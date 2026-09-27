# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Medium-mask aware dataset and datamanager.

Upstream Nerfstudio has no notion of a per-pixel medium. This module reads the medium mask
next to every image, folds the image alpha into a validity mask, and hands the medium mask
to the model on every ray bundle. Without it the model falls back to intersecting every ray
with the geometric water plane, and the numbers change silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Dict, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import pil_to_numpy
from PIL import Image

MEDIUM_MASK_KEY = "medium_mask"


def attach_medium_mask(ray_bundle: RayBundle, medium_mask: Optional[Any]) -> RayBundle:
    """Store ``medium_mask`` as ``(*ray_bundle.shape, 1)`` in the ray metadata, where the model reads it.

    ``None`` leaves the bundle untouched. The dtype is kept; :func:`read_medium_mask` thresholds it.
    """
    if medium_mask is None:
        return ray_bundle
    mask = torch.as_tensor(medium_mask).to(ray_bundle.origins.device)
    if mask.numel() != len(ray_bundle):
        raise ValueError(f"Medium mask size {mask.numel()} does not match number of rays {len(ray_bundle)}.")
    metadata = dict(ray_bundle.metadata) if ray_bundle.metadata is not None else {}
    metadata[MEDIUM_MASK_KEY] = mask.reshape(*ray_bundle.shape, 1)
    ray_bundle.metadata = metadata
    return ray_bundle


def read_medium_mask(ray_bundle: RayBundle, threshold: float) -> Optional[torch.Tensor]:
    """Flat ``(N,)`` water mask of the bundle (mask value above ``threshold``), or ``None``.

    ``None`` means "no medium mask, do not restrict". An all-False tensor would mean the
    opposite ("no ray is water"): a run without medium masks would train on the geometric
    water plane and then silently export an empty point cloud.
    """
    if not ray_bundle.metadata or MEDIUM_MASK_KEY not in ray_bundle.metadata:
        return None
    mask = ray_bundle.metadata[MEDIUM_MASK_KEY].to(ray_bundle.origins.device).reshape(-1).to(torch.float32)
    if mask.numel() != len(ray_bundle):
        raise ValueError(f"Medium mask size {mask.numel()} does not match number of rays {len(ray_bundle)}.")
    return mask > float(threshold)


def _open_resized(filepath: Union[Path, IO[bytes]], scale_factor: float) -> Image.Image:
    image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = image.size
        image = image.resize((int(width * scale_factor), int(height * scale_factor)), resample=Image.Resampling.NEAREST)
    return image


def _read_mask_array(filepath: Union[Path, IO[bytes]], scale_factor: float) -> np.ndarray:
    return pil_to_numpy(_open_resized(filepath, scale_factor))


def _mask_from_array(mask_np: np.ndarray) -> torch.Tensor:
    """Boolean ``[H, W, 1]`` mask: a pixel is set if any colour channel is non-zero."""
    if mask_np.ndim == 2:
        mask_np = mask_np[..., None]
    elif mask_np.ndim == 3:
        if mask_np.shape[-1] == 4:
            mask_np = mask_np[..., :3].astype(np.float32).mean(axis=-1, keepdims=True)
        elif mask_np.shape[-1] != 1:
            mask_np = mask_np.astype(np.float32).mean(axis=-1, keepdims=True)
    else:
        raise ValueError("Unsupported mask dimensions")

    mask_np = (mask_np != 0).astype(np.bool_)
    return torch.from_numpy(mask_np)


def _alpha_from_array(mask_np: np.ndarray) -> torch.Tensor:
    """Boolean ``[H, W, 1]`` validity from the alpha channel (all valid without one)."""
    if mask_np.ndim == 3 and mask_np.shape[-1] == 4:
        alpha_channel = mask_np[..., 3:4]
    else:
        alpha_channel = np.ones((*mask_np.shape[:2], 1), dtype=np.uint8) * 255

    alpha_bool = (alpha_channel > 127).astype(np.bool_)
    return torch.from_numpy(alpha_bool)


def load_mask(filepath: Union[Path, IO[bytes]], scale_factor: float = 1.0) -> torch.Tensor:
    """Read a mask image as a boolean ``[H, W, 1]`` tensor: a pixel is set if any colour channel is non-zero.

    Metashape exports masks as grey, RGB or RGBA PNGs. Upstream's
    ``get_image_mask_tensor_from_path`` accepts single-channel masks only, so RGB and RGBA masks
    are reduced here (the alpha channel of an RGBA mask is ignored).
    """
    return _mask_from_array(_read_mask_array(filepath, scale_factor))


def load_mask_alpha(filepath: Union[Path, IO[bytes]], scale_factor: float = 1.0) -> torch.Tensor:
    """Extract an alpha-channel validity mask from an image."""
    return _alpha_from_array(_read_mask_array(filepath, scale_factor))


class BathyInputDataset(InputDataset):
    """InputDataset that adds `medium_mask` and `image_valid_mask` to every sample."""

    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, cache_compressed_images: bool = False
    ):
        super().__init__(dataparser_outputs, scale_factor, cache_compressed_images)
        self.medium_mask_filenames = self.metadata.get("medium_mask_filenames")
        # Opt-in: promote the image alpha into the ray-sampling mask (`data["mask"]`). Off by
        # default. Alpha always enters `image_valid_mask` and `medium_mask` via get_metadata.
        self.use_alpha_as_sampling_mask = bool(dataparser_outputs.metadata.get("use_alpha_as_sampling_mask", False))

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        numpy_image = self.get_numpy_image(image_idx)
        alpha_mask = None
        if numpy_image.shape[-1] == 4:
            alpha_channel = numpy_image[..., 3:4]
            alpha_mask_np = (alpha_channel > 5).astype(np.bool_)
            alpha_mask = torch.from_numpy(alpha_mask_np).bool()

        if image_type == "float32":
            image = torch.from_numpy(numpy_image.astype(np.float32) / np.float32(255.0))
            if image.shape[-1] == 4:
                alpha = image[..., 3:]
                rgb = image[..., :3]
                if self._dataparser_outputs.alpha_color is not None:
                    alpha_color = torch.tensor(self._dataparser_outputs.alpha_color, dtype=image.dtype)
                    image = rgb * alpha + alpha_color.view(1, 1, 3) * (1.0 - alpha)
                else:
                    image = rgb
        elif image_type == "uint8":
            image = torch.from_numpy(numpy_image.copy())
            if image.shape[-1] == 4:
                if self._dataparser_outputs.alpha_color is not None:
                    alpha = image[..., 3:].float() / 255.0
                    rgb = image[..., :3].float()
                    alpha_color = torch.tensor(self._dataparser_outputs.alpha_color, dtype=torch.float32)
                    image = rgb * alpha + 255.0 * alpha_color.view(1, 1, 3) * (1.0 - alpha)
                    image = torch.clamp(image, min=0, max=255).to(torch.uint8)
                else:
                    image = image[..., :3]
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data: Dict[str, Any] = {"image_idx": image_idx, "image": image}
        if alpha_mask is not None:
            # `_alpha_mask` is the internal hand-off to `get_metadata`, which turns it into
            # `image_valid_mask` and ANDs it into `medium_mask`. That path always runs.
            # Promoting alpha into `data["mask"]` — the RAY SAMPLING mask — is opt-in.
            if self.use_alpha_as_sampling_mask:
                data["mask"] = alpha_mask
            data["_alpha_mask"] = alpha_mask
        if self._dataparser_outputs.mask_filenames is not None:
            if self.cache_compressed_images:
                mask_filepath = self.binary_masks[image_idx]
            else:
                mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            loaded_mask = load_mask(filepath=mask_filepath, scale_factor=self.scale_factor)
            if "mask" in data:
                data["mask"] = data["mask"] & loaded_mask
            else:
                data["mask"] = loaded_mask
            assert data["mask"].shape[:2] == data["image"].shape[:2], (
                f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
            )
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        data.pop("_alpha_mask", None)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        metadata: Dict[str, Any] = {}
        if self.medium_mask_filenames is None:
            return metadata
        mask_filename = self.medium_mask_filenames[data["image_idx"]]
        if mask_filename is None:
            return metadata
        # One decode serves both: the colour channels give the medium, the alpha the validity.
        mask_np = _read_mask_array(Path(mask_filename), self.scale_factor)
        medium_mask = _mask_from_array(mask_np)
        valid = _alpha_from_array(mask_np)
        alpha_mask = data.get("_alpha_mask")
        if alpha_mask is not None:
            valid = valid & alpha_mask
        metadata["image_valid_mask"] = valid
        metadata[MEDIUM_MASK_KEY] = medium_mask & valid
        return metadata


@dataclass
class BathyDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: BathyDataManager)


class BathyDataManager(VanillaDataManager[BathyInputDataset]):
    """VanillaDataManager whose datasets carry medium masks and whose ray bundles forward them."""

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        return attach_medium_mask(ray_bundle, batch.get(MEDIUM_MASK_KEY)), batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)
        return attach_medium_mask(ray_bundle, batch.get(MEDIUM_MASK_KEY)), batch
