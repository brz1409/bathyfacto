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

"""Metashape XML parsing for the BathyNerf dataset builder: chunk transform, markers, camera intrinsics/extrinsics, mask discovery, image downscaling."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional
from xml.etree.ElementTree import Element

import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image


def _require_attr(element: Element, name: str) -> str:
    """Return the named attribute value or raise ValueError if absent."""
    value = element.get(name)
    if value is None:
        raise ValueError(f"Expected attribute '{name}' in <{element.tag}>.")
    return value


def _require_child_text(element: Element, child_path: str) -> str:
    """Return the text of a child element or raise ValueError if absent."""
    child = element.find(child_path)
    if child is None or child.text is None:
        raise ValueError(f"Expected text node '{child_path}' in <{element.tag}>.")
    return child.text


def parse_metashape_xml(xml_file_path: Path) -> Element:
    """Parse a Metashape XML export once; the ``extract_*`` helpers read the returned root."""
    return ET.parse(xml_file_path).getroot()


def extract_chunk_transform(root: Element) -> tuple[float, np.ndarray, np.ndarray]:
    """Extract the global chunk transformation (scale, rotation, translation) from a Metashape XML file.

    Args:
        root: Root of the parsed Metashape XML (see :func:`parse_metashape_xml`).

    Returns:
        Tuple of (scale, rotation 3x3, translation 3-vector). Returns (1.0, I, 0) when
        no ``<chunk><transform>`` block is present (non-georeferenced dataset).
    """
    transform_node = root.find("./chunk/transform")

    if transform_node is not None:
        scale_str = _require_child_text(transform_node, "scale")
        rotation_str = _require_child_text(transform_node, "rotation")
        translation_str = _require_child_text(transform_node, "translation")

        s = float(scale_str)
        R = np.array(list(map(float, rotation_str.split()))).reshape((3, 3))
        T = np.array(list(map(float, translation_str.split())))
        return s, R, T
    return 1.0, np.identity(3), np.zeros(3)


def extract_marker_coordinates(root: Element) -> dict[str, np.ndarray]:
    """Extract marker labels and their 3D global coordinates from a Metashape XML file.

    Args:
        root: Root of the parsed Metashape XML (see :func:`parse_metashape_xml`).

    Returns:
        Dict mapping marker label -> (3,) global coordinate array.
    """
    marker_coordinates: dict[str, np.ndarray] = {}
    for marker in root.findall("./chunk/markers/marker"):
        marker_label = _require_attr(marker, "label")
        reference = marker.find("reference")
        if reference is not None:
            x = float(_require_attr(reference, "x"))
            y = float(_require_attr(reference, "y"))
            z = float(_require_attr(reference, "z"))
            marker_coordinates[marker_label] = np.array([x, y, z])
    return marker_coordinates


def extract_camera_intrinsics(root: Element) -> dict[str, Any]:
    """Extract camera intrinsics and image dimensions from the XML file.

    Assumes all cameras share the same sensor and calibration.

    Args:
        root: Root of the parsed Metashape XML (see :func:`parse_metashape_xml`).

    Returns:
        Dict with keys: fx, fy, cx, cy, width, height, distortion_params.

    Raises:
        ValueError: If calibration or resolution nodes are absent.
    """
    calibration_node = root.find(".//sensor/calibration")
    resolution_node = root.find(".//sensor/resolution")
    if calibration_node is None or resolution_node is None:
        raise ValueError("Could not find camera calibration or resolution in XML.")

    f = float(_require_child_text(calibration_node, "f"))

    # Missing calibration entries default to 0.0 (e.g. an initial calibration).
    def get_calib_val(name: str) -> float:
        node = calibration_node.find(name)
        return float(node.text) if node is not None and node.text is not None else 0.0

    width = int(_require_attr(resolution_node, "width"))
    height = int(_require_attr(resolution_node, "height"))

    return {
        "fx": f,
        "fy": f,
        "cx": get_calib_val("cx") + width / 2.0,
        "cy": get_calib_val("cy") + height / 2.0,
        "width": width,
        "height": height,
        # Nerfstudio expects 6 distortion parameters: k1, k2, k3, k4, p1, p2.
        "distortion_params": [get_calib_val(name) for name in ("k1", "k2", "k3", "k4", "p1", "p2")],
    }


def extract_camera_transforms(root: Element, image_extension: str = ".png") -> tuple[np.ndarray, list[str]]:
    """Extract 4x4 transformation matrices for each camera, sorted by label.

    Applies the Metashape→Nerfstudio coordinate convention flip (180° around X-axis).

    Args:
        root: Root of the parsed Metashape XML (see :func:`parse_metashape_xml`).
        image_extension: File extension for images (default: ".png").

    Returns:
        Tuple of (camera_to_worlds (N, 4, 4), image_filenames list).
    """
    cameras = root.findall(".//camera")
    cameras_sorted = sorted(cameras, key=lambda cam: cam.get("label", ""))

    camera_transforms = []
    image_filenames = []

    for camera in cameras_sorted:
        label = camera.get("label", "unknown_camera")
        transform_elem = camera.find("transform")

        if transform_elem is None or transform_elem.text is None:
            CONSOLE.log(f"[yellow]bathyfacto  dataset: skipping camera {label}, no transform data[/yellow]")
            continue

        transform_str = transform_elem.text
        transform_values = list(map(float, transform_str.split()))
        transform_matrix = np.array(transform_values).reshape((4, 4))

        # Nerfstudio uses a different coordinate system convention (OpenCV/COLMAP).
        # The camera poses are flipped [x, y, z] -> [x, -y, -z],
        # a 180-degree rotation around the X-axis.
        transform_matrix = transform_matrix @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        camera_transforms.append(transform_matrix)
        # Assume images are in an 'images' folder and have the same name as the camera label
        image_filenames.append(f"images/{label}{image_extension}")

    return np.array(camera_transforms), image_filenames


def match_mask_filenames_to_images(
    image_filenames: list[str],
    mask_dir: Path,
    mask_extensions: tuple[str, ...] = (".png", ".jpg", ".JPG", ".jpeg"),
) -> list[Optional[str]]:
    """Match medium mask files to image filenames by stem.

    Args:
        image_filenames: List of image paths (e.g., ['images/0001.png', ...])
        mask_dir: Directory containing mask files
        mask_extensions: Tuple of valid mask file extensions

    Returns:
        List of mask paths matching the image_filenames order, or None for unmatched images.

    Raises:
        RuntimeError: If mask_dir does not exist or no masks were matched.
    """
    if not mask_dir.exists():
        raise RuntimeError(f"Mask directory {mask_dir} does not exist.")

    mask_map: dict[str, Path] = {}
    for ext in mask_extensions:
        for mask_path in mask_dir.glob(f"*{ext}"):
            stem = mask_path.stem
            if stem not in mask_map:
                mask_map[stem] = mask_path

    matched_masks: list[Optional[str]] = []
    num_matched = 0
    for img_filename in image_filenames:
        img_stem = Path(img_filename).stem

        if img_stem in mask_map:
            matched_masks.append(str(mask_map[img_stem].relative_to(mask_dir.parent)))
            num_matched += 1
        else:
            matched_masks.append(None)

    CONSOLE.log(
        f"bathyfacto  dataset: {len(mask_map)} mask files found in {mask_dir}, "
        f"{num_matched}/{len(image_filenames)} images matched by filename stem"
    )

    if num_matched == 0:
        raise RuntimeError("No masks were matched to images. Check that mask filenames match image filenames.")

    return matched_masks


def scale_intrinsics(intrinsics: dict[str, Any], scale_factor: int) -> dict[str, Any]:
    """Scale camera intrinsics by a given downscale factor.

    Args:
        intrinsics: Original intrinsics dict with 'fx', 'fy', 'cx', 'cy', 'width', 'height'.
        scale_factor: Downscale factor (2, 4, 8, etc.)

    Returns:
        New intrinsics dict with scaled values.
    """
    scaled_intrinsics = copy.deepcopy(intrinsics)
    scaled_intrinsics["fx"] = intrinsics["fx"] / scale_factor
    scaled_intrinsics["fy"] = intrinsics["fy"] / scale_factor
    scaled_intrinsics["cx"] = intrinsics["cx"] / scale_factor
    scaled_intrinsics["cy"] = intrinsics["cy"] / scale_factor
    scaled_intrinsics["width"] = intrinsics["width"] // scale_factor
    scaled_intrinsics["height"] = intrinsics["height"] // scale_factor
    return scaled_intrinsics


def downscale_images_and_masks(
    image_dir: Path,
    mask_dir: Optional[Path],
    output_base_dir: Path,
    num_downscales: int,
    verbose: bool = False,
    *,
    xml_root: Optional[Element],
) -> tuple[dict[int, list[str]], dict[int, Optional[list[Optional[str]]]], dict[int, tuple[int, int]]]:
    """Downscale images and corresponding masks to multiple resolutions.

    Args:
        image_dir: Directory containing the original images (e.g., 'images')
        mask_dir: Directory containing the original masks (e.g., 'medium_masks'), or None
        output_base_dir: Base output directory where scaled subdirectories will be created
        num_downscales: Number of downscale levels (each level downscales by 2x), at least 1
        verbose: Print progress messages
        xml_root: Root of the Metashape XML the poses were read from. Its aligned cameras,
            sorted by label, define the image list, so images and poses stay parallel arrays.
            ``None`` takes every image on disk, sorted by name.

    Returns:
        Tuple of:
        - downscaled_image_filenames: dict[scale_level] -> list of relative image paths
        - downscaled_mask_filenames: dict[scale_level] -> list of relative mask paths (or None)
        - image_dimensions: dict[scale_level] -> (height, width)
    """
    if num_downscales < 1:
        raise ValueError(f"num_downscales must be at least 1, got {num_downscales}.")

    valid_stems: list[str] = []
    if xml_root is not None:
        cameras = sorted(xml_root.findall(".//camera"), key=lambda cam: cam.get("label", ""))

        for cam in cameras:
            transform = cam.find("transform")
            if transform is not None and transform.text is not None:
                label = cam.get("label")
                if label is not None:
                    valid_stems.append(label)

    image_map = {f.stem: f for f in image_dir.glob("*") if f.is_file()}

    # Rebuild the image list strictly matching the XML order and avoiding unaligned cameras
    if valid_stems:
        original_images = [image_map[stem] for stem in valid_stems if stem in image_map]
    else:
        # No XML, or no aligned camera in it: every image on disk, sorted by name.
        original_images = sorted(list(image_map.values()), key=lambda x: x.name)

    downscaled_image_filenames: dict[int, list[str]] = {}
    downscaled_mask_filenames: dict[int, Optional[list[Optional[str]]]] = {}
    image_dimensions: dict[int, tuple[int, int]] = {}

    # `original_images` is already label-sorted and filtered to aligned cameras (see above).
    # Re-globbing it here would undo the filter and break pose correspondence.
    if not original_images:
        raise ValueError(f"No images found in {image_dir}")

    mask_map_by_stem: dict[str, Path] = {}
    if mask_dir and mask_dir.exists():
        for mask in sorted(f for f in mask_dir.glob("*") if f.is_file()):
            mask_map_by_stem[mask.stem] = mask

    if verbose:
        level_word = "level" if num_downscales == 1 else "levels"
        CONSOLE.log(f"bathyfacto  dataset: downscaling {len(original_images)} images to {num_downscales} {level_word}")

    with Image.open(original_images[0]) as img:
        orig_height, orig_width = img.size[1], img.size[0]  # PIL: (width, height)

    for scale_level in range(1, num_downscales + 1):
        scale_factor = 2**scale_level  # 2x, 4x, 8x, etc.

        scaled_image_dir = output_base_dir / f"images_{scale_factor}"
        scaled_image_dir.mkdir(parents=True, exist_ok=True)

        scaled_mask_dir_out = None
        if mask_map_by_stem:
            scaled_mask_dir_out = output_base_dir / f"medium_masks_{scale_factor}"
            scaled_mask_dir_out.mkdir(parents=True, exist_ok=True)

        scaled_image_paths: list[str] = []
        scaled_mask_paths: Optional[list[Optional[str]]] = [] if mask_map_by_stem else None

        new_height = orig_height // scale_factor
        new_width = orig_width // scale_factor

        for orig_image in original_images:
            with Image.open(orig_image) as img:
                scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                output_image_path = scaled_image_dir / orig_image.name
                scaled_img.save(output_image_path, quality=95)
                scaled_image_paths.append(f"images_{scale_factor}/{orig_image.name}")

                # Downscale corresponding mask if it exists. Always append exactly one entry
                # per image (`None` when this image has no mask), so the mask list stays a
                # true parallel array of the image list. Skipping the append would shorten
                # the list and shift every later image onto the wrong mask.
                if scaled_mask_dir_out is not None and scaled_mask_paths is not None:
                    mask_file = mask_map_by_stem.get(orig_image.stem)
                    if mask_file:
                        with Image.open(mask_file) as mask_img:
                            scaled_mask = mask_img.resize((new_width, new_height), Image.Resampling.NEAREST)
                            output_mask_path = scaled_mask_dir_out / mask_file.name
                            scaled_mask.save(output_mask_path)
                            scaled_mask_paths.append(f"medium_masks_{scale_factor}/{mask_file.name}")
                    else:
                        scaled_mask_paths.append(None)

        downscaled_image_filenames[scale_factor] = scaled_image_paths
        downscaled_mask_filenames[scale_factor] = scaled_mask_paths
        image_dimensions[scale_factor] = (new_height, new_width)

        if verbose:
            num_masks = len(scaled_mask_paths) if scaled_mask_paths is not None else 0
            CONSOLE.log(
                f"bathyfacto  dataset: {scale_factor}x downscale, {len(scaled_image_paths)} images, "
                f"{num_masks} masks, written to {scaled_image_dir}"
            )

    return downscaled_image_filenames, downscaled_mask_filenames, image_dimensions
