# Copyright 2026 Markus Brezovsky, TU Wien
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end downscaling: ``bathyfacto.build_dataset`` with ``--num-downscales 1``, then
``BathyNerfDataParserConfig(downscale_factor=2)``.

The builder's ``train_2x.npz`` already points into ``images_2/`` and carries halved intrinsics,
so the dataparser must load it as is. Remapping its paths again would give ``images_2_2/...``
(missing files), and rescaling its intrinsics again would divide them by 4 instead of 2.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bathyfacto.bathyfacto_dataparser import BathyNerfDataParser, BathyNerfDataParserConfig
from bathyfacto.build_dataset import main as build_main

IMG_SIZE = 16
N_CAMERAS = 8
N_MARKERS = 6


def _write_markers_xml(root: Path, *, xml_name: str = "markers.xml", unaligned_label: str = "") -> None:
    """A tiny synthetic Metashape XML: valid sensor, N_CAMERAS cameras, N_MARKERS markers.

    ``unaligned_label`` adds a camera without a ``<transform>`` (not aligned by Metashape) that
    still has an image and a mask on disk.
    """
    image_dir = root / "images"
    mask_dir = root / "medium_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cameras_xml = []
    for i in range(N_CAMERAS):
        label = f"{i:04d}"
        angle = 2 * math.pi * i / N_CAMERAS
        x, y, z = 5.0 * math.cos(angle), 5.0 * math.sin(angle), 8.0 + 0.2 * (i % 3)
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]
        transform_str = " ".join(f"{v:.6f}" for v in transform.flatten())
        cameras_xml.append(f'      <camera id="{i}" label="{label}"><transform>{transform_str}</transform></camera>')

        Image.fromarray(np.full((IMG_SIZE, IMG_SIZE, 3), 120, dtype=np.uint8), mode="RGB").save(
            image_dir / f"{label}.png"
        )
        Image.fromarray(np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8), mode="L").save(mask_dir / f"{label}.png")

    if unaligned_label:
        cameras_xml.append(f'      <camera id="{N_CAMERAS}" label="{unaligned_label}"/>')
        Image.fromarray(np.full((IMG_SIZE, IMG_SIZE, 3), 30, dtype=np.uint8), mode="RGB").save(
            image_dir / f"{unaligned_label}.png"
        )
        Image.fromarray(np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8), mode="L").save(
            mask_dir / f"{unaligned_label}.png"
        )

    rng = np.random.default_rng(0)
    markers_xml = []
    for i in range(N_MARKERS):
        angle = 2 * math.pi * i / N_MARKERS
        mx, my = 3.0 * math.cos(angle), 3.0 * math.sin(angle)
        mz = float(rng.normal(0.0, 0.01))
        markers_xml.append(
            f'      <marker label="marker{i}"><reference x="{mx:.4f}" y="{my:.4f}" z="{mz:.4f}"/></marker>'
        )

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <sensors>
      <sensor>
        <resolution width="{IMG_SIZE}" height="{IMG_SIZE}"/>
        <calibration>
          <f>20.0</f>
        </calibration>
      </sensor>
    </sensors>
    <cameras>
{chr(10).join(cameras_xml)}
    </cameras>
    <markers>
{chr(10).join(markers_xml)}
    </markers>
  </chunk>
</document>
"""
    (root / xml_name).write_text(xml)


def _build_args(data_dir: Path, *, num_downscales: int, xml_filename: str = "markers.xml") -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=data_dir,
        medium_mask_dir=Path("medium_masks"),
        aabb_scale=1.0,
        aabb_center="auto",
        underwater_scene=False,
        scene_box_from_mesh=False,
        scene_box_buffer_m=0.5,
        xml_filename=xml_filename,
        image_extension=".png",
        num_downscales=num_downscales,
    )


@pytest.fixture
def built_dataset(tmp_path) -> Path:
    _write_markers_xml(tmp_path)
    build_main(_build_args(tmp_path, num_downscales=1))
    return tmp_path


def test_prescaled_split_is_preferred_and_paths_exist(built_dataset):
    config = BathyNerfDataParserConfig(data=built_dataset, downscale_factor=2)
    parser = BathyNerfDataParser(config)
    outputs = parser.get_dataparser_outputs(split="train")

    assert len(outputs.image_filenames) > 0
    for image_path in outputs.image_filenames:
        assert image_path.exists(), f"{image_path} does not exist"
        assert image_path.parent.name == "images_2", f"{image_path} not under images_2/"


def test_downscaled_image_sizes_match_camera_width_height(built_dataset):
    config = BathyNerfDataParserConfig(data=built_dataset, downscale_factor=2)
    parser = BathyNerfDataParser(config)
    outputs = parser.get_dataparser_outputs(split="train")

    expected_size = IMG_SIZE // 2
    heights = outputs.cameras.height.flatten().tolist()
    widths = outputs.cameras.width.flatten().tolist()
    assert all(h == expected_size for h in heights), heights
    assert all(w == expected_size for w in widths), widths

    with Image.open(outputs.image_filenames[0]) as img:
        assert img.size == (expected_size, expected_size)


def test_downscaled_intrinsics_are_exactly_half_not_a_quarter(built_dataset):
    """The regression this test pins: applying the dataparser's own halving on top of the
    builder's already-halved intrinsics would produce a quarter, not a half."""
    full_res_config = BathyNerfDataParserConfig(data=built_dataset)
    full_res_outputs = BathyNerfDataParser(full_res_config).get_dataparser_outputs(split="train")

    downscaled_config = BathyNerfDataParserConfig(data=built_dataset, downscale_factor=2)
    downscaled_outputs = BathyNerfDataParser(downscaled_config).get_dataparser_outputs(split="train")

    for attr in ("fx", "fy", "cx", "cy"):
        full_res = getattr(full_res_outputs.cameras, attr).flatten()
        downscaled = getattr(downscaled_outputs.cameras, attr).flatten()
        assert torch_allclose_half(full_res, downscaled), (
            f"{attr}: full-res {full_res.tolist()} vs downscaled {downscaled.tolist()}, "
            "expected downscaled == full_res / 2"
        )


def torch_allclose_half(full_res, downscaled) -> bool:
    import torch

    return bool(torch.allclose(downscaled, full_res / 2.0, atol=1e-3))


def test_full_resolution_load_is_unaffected_by_the_fix(built_dataset):
    """downscale_factor=None (the paper path) must not go anywhere near the new branch."""
    config = BathyNerfDataParserConfig(data=built_dataset)
    outputs = BathyNerfDataParser(config).get_dataparser_outputs(split="train")

    for image_path in outputs.image_filenames:
        assert image_path.parent.name == "images"
        with Image.open(image_path) as img:
            assert img.size == (IMG_SIZE, IMG_SIZE)


def test_downscaled_splits_follow_the_xml_named_by_xml_filename(tmp_path):
    """With ``--xml-filename`` the downscaler must filter by that XML, not by ``markers.xml``.

    The XML has a camera without a pose in the middle of the label order. The downscaled split
    must list exactly the images of the full-resolution split, in the same order, so every image
    keeps its own pose.
    """
    _write_markers_xml(tmp_path, xml_name="chunk_export.xml", unaligned_label="0003a")
    build_main(_build_args(tmp_path, num_downscales=1, xml_filename="chunk_export.xml"))

    full_res = np.load(tmp_path / "data.npz", allow_pickle=True)["image_filenames"]
    downscaled = np.load(tmp_path / "data_2x.npz", allow_pickle=True)["image_filenames"]
    assert [Path(p).stem for p in downscaled] == [Path(p).stem for p in full_res]
    assert "0003a" not in [Path(p).stem for p in downscaled]


def test_full_resolution_split_with_relative_data_dir_remaps_to_existing_files(built_dataset, monkeypatch):
    """A plain ``train.npz`` plus ``images_2/`` loaded with a relative ``--data`` and factor 2.

    The dataparser remaps ``images/`` to ``images_2/`` itself on this path. The remapped paths
    must point at the files on disk, with the dataset root prefixed once.
    """
    for prescaled in built_dataset.glob("*_2x.npz"):
        prescaled.unlink()
    monkeypatch.chdir(built_dataset.parent)
    relative_data = Path(built_dataset.name)

    outputs = BathyNerfDataParser(
        BathyNerfDataParserConfig(data=relative_data, downscale_factor=2)
    ).get_dataparser_outputs(split="train")

    for image_path in outputs.image_filenames:
        assert image_path == relative_data / "images_2" / image_path.name
        assert image_path.exists()
    for mask_path in outputs.metadata["medium_mask_filenames"]:
        assert mask_path == relative_data / "medium_masks_2" / mask_path.name
        assert mask_path.exists()


def test_test_split_falls_back_to_the_prescaled_val_split(built_dataset):
    """Without a test split, ``split="test"`` loads ``val_2x.npz``, which is already downscaled.

    Its paths must be used as they are. Remapping them again gives ``images_2_2/`` and fails.
    """
    for test_split in built_dataset.glob("test*.npz"):
        test_split.unlink()

    outputs = BathyNerfDataParser(
        BathyNerfDataParserConfig(data=built_dataset, downscale_factor=2)
    ).get_dataparser_outputs(split="test")

    expected_size = IMG_SIZE // 2
    assert len(outputs.image_filenames) > 0
    for image_path in outputs.image_filenames:
        assert image_path.parent.name == "images_2", f"{image_path} not under images_2/"
        with Image.open(image_path) as img:
            assert img.size == (expected_size, expected_size)
    assert outputs.cameras.width.flatten().tolist() == [expected_size] * len(outputs.image_filenames)
