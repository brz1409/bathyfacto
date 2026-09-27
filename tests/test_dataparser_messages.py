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

"""BathyNerfDataParser console messages: one summary line per split, no chatter.

Messages are captured by swapping in a recorder for the module's ``CONSOLE`` rather than
reading terminal output, so rich's auto-highlighting and line wrapping cannot corrupt the
assertions (a real terminal run of these same fixtures wraps the summary line across two
rows and colors the plane height and scale as separate spans).
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

import bathyfacto.bathyfacto_dataparser as dataparser_module
from bathyfacto.bathyfacto_dataparser import (
    BathyNerfDataParser,
    BathyNerfDataParserConfig,
    _water_plane_and_scale_suffix,
)

_HEIGHT, _WIDTH = 4, 6


@pytest.fixture
def recorder(record_console):
    return record_console(dataparser_module)


def _write_dataset(
    data_dir: Path,
    split: str,
    *,
    num_images: int = 2,
    with_medium_masks: bool = True,
    with_water_surface: bool = True,
) -> None:
    """Write a minimal train/val/test NPZ split with just enough keys for the dataparser."""
    image_filenames = np.array([f"images/{i:04d}.png" for i in range(num_images)])
    cameras: Dict[str, np.ndarray] = {
        "fx": np.full((num_images, 1), 100.0, dtype=np.float32),
        "fy": np.full((num_images, 1), 100.0, dtype=np.float32),
        "cx": np.full((num_images, 1), _WIDTH / 2, dtype=np.float32),
        "cy": np.full((num_images, 1), _HEIGHT / 2, dtype=np.float32),
        "height": np.full((num_images, 1), _HEIGHT, dtype=np.int64),
        "width": np.full((num_images, 1), _WIDTH, dtype=np.int64),
        "camera_to_worlds": np.tile(np.eye(4, dtype=np.float32), (num_images, 1, 1)),
        "camera_type": np.ones((num_images, 1), dtype=np.int64),
    }
    scene_box = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    metadata: Dict[str, Any] = {}
    if with_medium_masks:
        metadata["medium_mask_filenames"] = [f"medium_masks/{i:04d}.png" for i in range(num_images)]
    if with_water_surface:
        # d = 0.6165167689323425 matches a real bathy training dataset's train-split water
        # plane (z = -0.617 in scene coordinates).
        metadata["water_surface"] = {
            "plane_model": {"normal": [0.0, 0.0, 1.0], "d": 0.6165167689323425},
            "source": "test fixture",
        }

    np.savez(
        data_dir / f"{split}.npz",
        image_filenames=image_filenames,
        cameras=np.array(cameras, dtype=object),
        scene_box=scene_box,
        metadata=np.array(metadata, dtype=object),
        normalization_scale=np.float32(12.776531219482422),
        applied_scale=np.float32(4.7607293128967285),
    )


def _parser(data_dir: Path) -> BathyNerfDataParser:
    return BathyNerfDataParser(BathyNerfDataParserConfig(data=data_dir))


def test_train_summary_line_reports_images_masks_water_plane_and_scale(tmp_path, recorder):
    _write_dataset(tmp_path, "train", num_images=2)

    _parser(tmp_path)._generate_dataparser_outputs(split="train")

    assert recorder.messages == [
        "bathyfacto  train: 2 images, 2/2 medium masks, water plane z=-0.617 (scene), 60.83 m per scene unit"
    ]


def test_val_summary_line_has_no_water_plane_or_scale(tmp_path, recorder):
    _write_dataset(tmp_path, "val", num_images=3, with_water_surface=True)

    _parser(tmp_path)._generate_dataparser_outputs(split="val")

    assert recorder.messages == ["bathyfacto  val: 3 images, 3/3 medium masks"]
    assert "water plane" not in recorder.messages[0]
    assert "per scene unit" not in recorder.messages[0]


def test_no_medium_mask_directory_is_exactly_one_yellow_line_per_split(tmp_path, recorder):
    """No `medium_masks/` dir and no masks recorded in the NPZ: one line naming the image count,
    the checked directory and the geometric-plane fallback."""
    _write_dataset(tmp_path, "train", num_images=2, with_medium_masks=False)

    _parser(tmp_path)._generate_dataparser_outputs(split="train")

    expected_dir = tmp_path / "medium_masks"
    assert recorder.messages == [
        f"[yellow]bathyfacto  train: 2 images, no medium masks in {expected_dir}, "
        "water comes from the geometric plane.[/yellow]"
    ]


def test_no_medium_mask_directory_val_line_has_no_water_plane_or_scale(tmp_path, recorder):
    """The no-mask line follows the same train-only water-plane/scale rule as the success line."""
    _write_dataset(tmp_path, "val", num_images=3, with_medium_masks=False)

    _parser(tmp_path)._generate_dataparser_outputs(split="val")

    assert len(recorder.messages) == 1
    assert "water plane" not in recorder.messages[0]
    assert "per scene unit" not in recorder.messages[0]


def test_test_split_fallback_uses_the_harmonized_prefix(tmp_path, recorder):
    """The test->val fallback line follows the same `bathyfacto  <split>:` prefix as everything else."""
    (tmp_path / "val.npz").write_bytes(b"")
    parser = _parser(tmp_path)

    result = parser._resolve_split_filepath("test")

    assert result == (tmp_path / "val.npz", False)
    assert recorder.messages == ["[yellow]bathyfacto  test: no test split, using val.npz[/yellow]"]


def test_water_plane_and_scale_suffix_reads_existing_metadata_without_new_physics():
    metadata = {
        "water_surface": {"plane_model": {"normal": [0.0, 0.0, 1.0], "d": 0.5}},
        "eff_scale": 60.868,
    }
    assert _water_plane_and_scale_suffix(metadata) == ", water plane z=-0.500 (scene), 60.87 m per scene unit"


def test_water_plane_and_scale_suffix_is_empty_without_water_surface_metadata():
    assert _water_plane_and_scale_suffix({"eff_scale": 1.0}) == ", 1.00 m per scene unit"
    assert _water_plane_and_scale_suffix({}) == ""
