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

"""Medium-mask contract of BathyInputDataset and BathyDataManager (paper behaviour)."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.pixel_samplers import PixelSamplerConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from PIL import Image

from bathyfacto.bathyfacto_datamanager import (
    BathyDataManager,
    BathyInputDataset,
    attach_medium_mask,
    load_mask,
    load_mask_alpha,
)

_H, _W = 4, 6


def _write_rgba(path: Path, *, height: int = _H, width: int = _W) -> None:
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = 128
    rgba[..., 3] = 255
    rgba[0, 0, 3] = 0  # one transparent pixel
    Image.fromarray(rgba, mode="RGBA").save(path)


def _write_gray_mask(path: Path, *, height: int = _H, width: int = _W) -> None:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 2 :, :] = 255  # bottom half water
    Image.fromarray(mask, mode="L").save(path)


def _outputs(
    tmp_path: Path,
    *,
    with_medium_mask: bool = True,
    sampling: bool = False,
    height: int = _H,
    width: int = _W,
) -> DataparserOutputs:
    image = tmp_path / "0001.png"
    _write_rgba(image, height=height, width=width)
    metadata: Dict[str, Any] = {"use_alpha_as_sampling_mask": sampling}
    if with_medium_mask:
        (tmp_path / "medium_masks").mkdir(exist_ok=True)
        medium = tmp_path / "medium_masks" / "0001.png"
        _write_gray_mask(medium, height=height, width=width)
        metadata["medium_mask_filenames"] = [medium]
    cameras = Cameras(
        camera_to_worlds=torch.eye(4)[:3, :4].unsqueeze(0),
        fx=torch.tensor([[1.0]]),
        fy=torch.tensor([[1.0]]),
        cx=torch.tensor([[width / 2]]),
        cy=torch.tensor([[height / 2]]),
        width=torch.tensor([[width]]),
        height=torch.tensor([[height]]),
        camera_type=CameraType.PERSPECTIVE,
    )
    return DataparserOutputs(
        image_filenames=[image],
        cameras=cameras,
        scene_box=SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])),
        metadata=metadata,
    )


def test_dataset_type_resolves_via_upstream_property():
    """Upstream's own resolver finds BathyInputDataset as the datamanager's dataset type."""
    assert BathyDataManager.__new__(BathyDataManager).dataset_type is BathyInputDataset


def test_alpha_is_not_a_sampling_mask_by_default(tmp_path):
    data = BathyInputDataset(_outputs(tmp_path)).get_data(0)
    assert "mask" not in data


def test_alpha_becomes_sampling_mask_when_opted_in(tmp_path):
    data = BathyInputDataset(_outputs(tmp_path, sampling=True)).get_data(0)
    assert not bool(data["mask"][0, 0, 0]) and bool(data["mask"][_H - 1, _W - 1, 0])


def test_medium_and_valid_mask(tmp_path):
    data = BathyInputDataset(_outputs(tmp_path)).get_data(0)
    assert data["image"].shape == (_H, _W, 3), "RGBA must be reduced to RGB when alpha_color is None"
    assert not bool(data["image_valid_mask"][0, 0, 0]), "image alpha > 5 is part of validity"
    assert not bool(data["medium_mask"][0, 0, 0]), "top half is air"
    assert bool(data["medium_mask"][_H - 1, 0, 0]), "bottom half is water"
    assert "_alpha_mask" not in data


def test_no_medium_mask_means_no_mask_keys(tmp_path):
    data = BathyInputDataset(_outputs(tmp_path, with_medium_mask=False)).get_data(0)
    assert "medium_mask" not in data and "image_valid_mask" not in data


def test_medium_mask_independent_of_sampling_flag(tmp_path):
    """`use_alpha_as_sampling_mask` only ever affects `data["mask"]`, never the medium mask."""
    off = BathyInputDataset(_outputs(tmp_path, sampling=False)).get_data(0)
    on = BathyInputDataset(_outputs(tmp_path, sampling=True)).get_data(0)
    assert torch.equal(off["medium_mask"], on["medium_mask"])
    assert torch.equal(off["image_valid_mask"], on["image_valid_mask"])


def test_uint8_path_matches_float32_masks(tmp_path):
    dataset = BathyInputDataset(_outputs(tmp_path))
    d32 = dataset.get_data(0, image_type="float32")
    d8 = dataset.get_data(0, image_type="uint8")
    assert d8["image"].shape == (_H, _W, 3)
    assert d8["image"].dtype == torch.uint8
    assert torch.equal(d8["medium_mask"], d32["medium_mask"])
    assert torch.equal(d8["image_valid_mask"], d32["image_valid_mask"])


def test_dataset_downscale_mask_matches_image_shape(tmp_path):
    """Dataset-level downscale (`scale_factor`, not the loader-level test above)."""
    outputs = _outputs(tmp_path, height=8, width=12)
    data = BathyInputDataset(outputs, scale_factor=0.5).get_data(0)
    assert data["image"].shape[:2] == (4, 6)
    assert data["medium_mask"].shape[:2] == data["image"].shape[:2]


@pytest.mark.parametrize("mode,pixel", [("L", 1), ("RGB", (0, 0, 7)), ("RGBA", (0, 3, 0, 255))])
def test_mask_threshold_any_channel_nonzero(tmp_path, mode, pixel):
    size = (2, 2)
    img = Image.new(mode, size, 0 if mode == "L" else tuple([0] * len(mode)))
    img.putpixel((1, 1), pixel)
    path = tmp_path / f"m_{mode}.png"
    img.save(path)
    mask = load_mask(path)
    assert mask.shape == (2, 2, 1) and mask.dtype == torch.bool
    assert bool(mask[1, 1, 0]) and not bool(mask[0, 0, 0])


def test_mask_alpha_threshold(tmp_path):
    rgba = np.zeros((1, 3, 4), dtype=np.uint8)
    rgba[0, :, 3] = [127, 128, 255]
    path = tmp_path / "a.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    assert load_mask_alpha(path)[0, :, 0].tolist() == [False, True, True]


def test_downscaled_mask_matches_image(tmp_path):
    big = np.zeros((8, 12), dtype=np.uint8)
    big[4:, :] = 255
    path = tmp_path / "big.png"
    Image.fromarray(big, mode="L").save(path)
    assert load_mask(path, scale_factor=0.5).shape == (4, 6, 1)


def test_attach_medium_mask_to_ray_bundle():
    bundle = RayBundle(origins=torch.zeros(5, 3), directions=torch.ones(5, 3), pixel_area=torch.ones(5, 1))
    batch = {"medium_mask": torch.ones(5, 1, dtype=torch.bool)}
    out = attach_medium_mask(bundle, batch.get("medium_mask"))
    assert out.metadata["medium_mask"].shape == (5, 1)
    no_mask = attach_medium_mask(
        RayBundle(origins=torch.zeros(5, 3), directions=torch.ones(5, 3), pixel_area=torch.ones(5, 1)), None
    )
    assert no_mask.metadata is None or "medium_mask" not in no_mask.metadata


def _wire_bare_datamanager(dataset: BathyInputDataset, batch: Dict, *, num_rays_per_batch: int = 8) -> BathyDataManager:
    """Builds a BathyDataManager through `__new__`, wiring only what next_train/next_eval touch.

    Skips the full `__init__` (trainer config, device, dataparser I/O) so the real
    `VanillaDataManager.next_train`/`next_eval` -> `PixelSampler.sample` -> `RayGenerator`
    chain can be exercised directly, with `attach_medium_mask` as the only bathy-specific step.
    """
    dm = BathyDataManager.__new__(BathyDataManager)
    dm.train_count = 0
    dm.eval_count = 0
    dm.iter_train_image_dataloader = iter([batch])
    dm.iter_eval_image_dataloader = iter([batch])
    dm.train_pixel_sampler = PixelSamplerConfig().setup(num_rays_per_batch=num_rays_per_batch)
    dm.eval_pixel_sampler = PixelSamplerConfig().setup(num_rays_per_batch=num_rays_per_batch)
    dm.train_ray_generator = RayGenerator(dataset.cameras)
    dm.eval_ray_generator = RayGenerator(dataset.cameras)
    return dm


@pytest.mark.parametrize("which", ["train", "eval"])
def test_next_train_and_next_eval_attach_through_real_chain(tmp_path, which):
    """next_train/next_eval attach the medium mask via the real CacheDataloader/PixelSampler/
    RayGenerator chain, not a hand-rolled shortcut, and the attached mask matches the dataset's
    own mask at the sampled (image, row, col) indices."""
    dataset = BathyInputDataset(_outputs(tmp_path))
    batch = nerfstudio_collate([dataset.get_data(0)])
    dm = _wire_bare_datamanager(dataset, batch)

    step_fn = dm.next_train if which == "train" else dm.next_eval
    ray_bundle, out_batch = step_fn(0)

    assert ray_bundle.metadata["medium_mask"].shape == (*ray_bundle.shape, 1)

    full_mask = dataset.get_data(0)["medium_mask"]  # (H, W, 1)
    y, x = out_batch["indices"][:, 1], out_batch["indices"][:, 2]
    expected = full_mask[y, x]
    assert torch.equal(ray_bundle.metadata["medium_mask"], expected)
    assert torch.equal(out_batch["medium_mask"], expected)


def test_sampling_indices_match_upstream_pixel_sampler(tmp_path):
    """With no medium mask and sampling opt-in off, BathyDataManager adds no `batch["mask"]`
    and its next_train chain draws exactly the same pixel indices as a plain upstream
    PixelSampler given the same seed and the same image batch."""
    dataset = BathyInputDataset(_outputs(tmp_path, with_medium_mask=False, sampling=False))
    batch = nerfstudio_collate([dataset.get_data(0)])
    assert "mask" not in batch

    dm = _wire_bare_datamanager(dataset, batch)
    torch.manual_seed(0)
    _, dm_batch = dm.next_train(0)

    plain_sampler = PixelSamplerConfig().setup(num_rays_per_batch=8)
    torch.manual_seed(0)
    plain_batch = plain_sampler.sample(batch)

    assert torch.equal(dm_batch["indices"], plain_batch["indices"])
