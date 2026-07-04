# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Utility functions to allow easy re-use of common operations across dataloaders"""

from pathlib import Path
from typing import IO, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage


def pil_to_numpy(im: PILImage) -> np.ndarray:
    """Converts a PIL Image object to a NumPy array.

    Args:
        im (PIL.Image.Image): The input PIL Image object.

    Returns:
        numpy.ndarray representing the image data.
    """
    # Load in image completely (PIL defaults to lazy loading)
    im.load()

    # Unpack data
    e = Image._getencoder(im.mode, "raw", im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        _, s, d = e.encode(bufsize)
        mem[offset : offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)

    return data


def get_image_mask_tensor_from_path(filepath: Union[Path, IO[bytes]], scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.Resampling.NEAREST)
    mask_np = pil_to_numpy(pil_mask)
    if mask_np.ndim == 2:
        mask_np = mask_np[..., None]
    elif mask_np.ndim == 3:
        if mask_np.shape[-1] == 1:
            pass
        elif mask_np.shape[-1] == 4:
            mask_np = mask_np[..., :3].astype(np.float32).mean(axis=-1, keepdims=True)
        else:
            mask_np = mask_np.astype(np.float32).mean(axis=-1, keepdims=True)
    else:
        raise ValueError("Unsupported mask dimensions")

    mask_np = (mask_np > 5).astype(np.bool_)
    mask_tensor = torch.from_numpy(mask_np)
    if mask_tensor.ndim != 3 or mask_tensor.shape[-1] != 1:
        raise ValueError("Mask tensor must have shape [H, W, 1]")
    return mask_tensor


def get_alpha_mask_from_path(filepath: Union[Path, IO[bytes]], scale_factor: float = 1.0) -> torch.Tensor:
    """Extract an alpha-channel validity mask from an image."""
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.Resampling.NEAREST)

    mask_np = pil_to_numpy(pil_mask)
    if mask_np.ndim == 3 and mask_np.shape[-1] == 4:
        alpha_channel = mask_np[..., 3:4]
    else:
        alpha_channel = np.ones((*mask_np.shape[:2], 1), dtype=np.uint8) * 255

    alpha_bool = (alpha_channel > 127).astype(np.bool_)
    alpha_tensor = torch.from_numpy(alpha_bool)
    if alpha_tensor.ndim != 3 or alpha_tensor.shape[-1] != 1:
        raise ValueError(f"Alpha mask tensor must have shape [H, W, 1], got {alpha_tensor.shape}")
    return alpha_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype=torch.int64).view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.Resampling.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath).astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)  # type: ignore
    return torch.from_numpy(image[:, :, np.newaxis])


def identity_collate(x):
    """This function does nothing but serves to help our dataloaders have a pickleable function, as lambdas are not pickleable"""
    return x
