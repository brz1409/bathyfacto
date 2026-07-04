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

"""Metashape XML parsing for the BathyNerf dataset builder: chunk transform, markers, camera intrinsics/extrinsics, mask discovery, image downscaling."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from xml.etree.ElementTree import Element

import numpy as np

from nerfstudio.utils.rich_utils import CONSOLE

if TYPE_CHECKING:
    pass


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


def extract_chunk_transform(xml_file_path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Extract the global chunk transformation (scale, rotation, translation) from a Metashape XML file.

    Args:
        xml_file_path: Path to the Metashape XML file.

    Returns:
        Tuple of (scale, rotation 3x3, translation 3-vector). Returns (1.0, I, 0) when
        no ``<chunk><transform>`` block is present (non-georeferenced dataset).
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
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


def extract_marker_coordinates(xml_file_path: Path) -> dict[str, np.ndarray]:
    """Extract marker labels and their 3D global coordinates from a Metashape XML file.

    Args:
        xml_file_path: Path to the Metashape XML file.

    Returns:
        Dict mapping marker label -> (3,) global coordinate array.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
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


def extract_camera_intrinsics(xml_file_path: Path) -> dict[str, Any]:
    """Extract camera intrinsics and image dimensions from the XML file.

    Assumes all cameras share the same sensor and calibration.

    Args:
        xml_file_path: Path to the Metashape XML file.

    Returns:
        Dict with keys: fx, fy, cx, cy, width, height, distortion_params.

    Raises:
        ValueError: If calibration or resolution nodes are absent.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    calibration_node = root.find(".//sensor/calibration")
    resolution_node = root.find(".//sensor/resolution")

    if calibration_node is not None and resolution_node is not None:
        f = float(_require_child_text(calibration_node, "f"))

        # Robustly extract cx/cy, defaulting to 0.0 if missing (e.g. initial calibration)
        cx_node = calibration_node.find("cx")
        cx = float(cx_node.text) if cx_node is not None and cx_node.text is not None else 0.0

        cy_node = calibration_node.find("cy")
        cy = float(cy_node.text) if cy_node is not None and cy_node.text is not None else 0.0

        width = int(_require_attr(resolution_node, "width"))
        height = int(_require_attr(resolution_node, "height"))

        # Extract distortion parameters if they exist, default to 0.0
        # Nerfstudio expects 6 parameters: k1, k2, k3, k4, p1, p2
        def get_calib_val(name: str) -> float:
            node = calibration_node.find(name)  # type: ignore[union-attr]
            return float(node.text) if node is not None and node.text is not None else 0.0

        k1 = get_calib_val("k1")
        k2 = get_calib_val("k2")
        k3 = get_calib_val("k3")
        k4 = get_calib_val("k4")
        p1 = get_calib_val("p1")
        p2 = get_calib_val("p2")

        intrinsics: dict[str, Any] = {
            "fx": f,
            "fy": f,
            "cx": cx + width / 2.0,
            "cy": cy + height / 2.0,
            "width": width,
            "height": height,
            "distortion_params": [k1, k2, k3, k4, p1, p2],
        }
        return intrinsics
    raise ValueError("Could not find camera calibration or resolution in XML.")


def extract_camera_transforms(xml_file_path: Path, image_extension: str = ".png") -> tuple[np.ndarray, list[str]]:
    """Extract 4x4 transformation matrices for each camera, sorted by label.

    Applies the Metashape→Nerfstudio coordinate convention flip (180° around X-axis).

    Args:
        xml_file_path: Path to the Metashape XML file.
        image_extension: File extension for images (default: ".png").

    Returns:
        Tuple of (camera_to_worlds (N, 4, 4), image_filenames list).
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    cameras = root.findall(".//camera")
    # Sort cameras by label to ensure consistent order
    cameras_sorted = sorted(cameras, key=lambda cam: cam.get("label", ""))

    camera_transforms = []
    image_filenames = []

    for camera in cameras_sorted:
        label = camera.get("label", "unknown_camera")
        transform_elem = camera.find("transform")

        if transform_elem is None or transform_elem.text is None:
            CONSOLE.log(f"[yellow]Warning: Skipping camera '{label}' because it is missing transform data.[/yellow]")
            continue  # Skip to the next camera

        transform_str = transform_elem.text
        transform_values = list(map(float, transform_str.split()))
        transform_matrix = np.array(transform_values).reshape((4, 4))

        # Nerfstudio uses a different coordinate system convention (OpenCV/COLMAP).
        # We need to apply a transformation to the camera poses.
        # [x, y, z] -> [x, -y, -z]
        # This corresponds to a 180-degree rotation around the X-axis.
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
    CONSOLE.log("[cyan]Mask naming rule: mask files must share the same filename stem as the image.[/cyan]")
    CONSOLE.log("[cyan]Example: images/DJI_..._0503.JPG -> masks/DJI_..._0503.png|jpg|jpeg[/cyan]")

    if not mask_dir.exists():
        raise RuntimeError(f"Mask directory {mask_dir} does not exist.")

    # Create a mapping of stem -> mask_path for all masks in the directory
    mask_map: dict[str, Path] = {}
    for ext in mask_extensions:
        for mask_path in mask_dir.glob(f"*{ext}"):
            stem = mask_path.stem
            if stem not in mask_map:
                mask_map[stem] = mask_path

    CONSOLE.log(f"[cyan]Found {len(mask_map)} mask files in {mask_dir}[/cyan]")

    # Match each image to its corresponding mask
    matched_masks: list[Optional[str]] = []
    num_matched = 0
    for img_filename in image_filenames:
        # Extract stem from image filename (e.g., 'images/0001.png' -> '0001')
        img_stem = Path(img_filename).stem

        if img_stem in mask_map:
            # Store relative path from data_dir
            matched_masks.append(str(mask_map[img_stem].relative_to(mask_dir.parent)))
            num_matched += 1
        else:
            matched_masks.append(None)

    CONSOLE.log(f"[cyan]Matched {num_matched}/{len(image_filenames)} images to masks[/cyan]")

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
) -> tuple[dict[int, list[str]], dict[int, Optional[list[Optional[str]]]], dict[int, tuple[int, int]]]:
    """Downscale images and corresponding masks to multiple resolutions.

    Args:
        image_dir: Directory containing the original images (e.g., 'images')
        mask_dir: Directory containing the original masks (e.g., 'medium_masks'), or None
        output_base_dir: Base output directory where scaled subdirectories will be created
        num_downscales: Number of downscale levels (each level downscales by 2x)
        verbose: Print progress messages

    Returns:
        Tuple of:
        - downscaled_image_filenames: dict[scale_level] -> list of relative image paths
        - downscaled_mask_filenames: dict[scale_level] -> list of relative mask paths (or None)
        - image_dimensions: dict[scale_level] -> (height, width)
    """
    from PIL import Image  # local import — PIL is optional/heavy

    # Filter: Find which cameras Metashape actually aligned by checking the XML
    valid_stems: list[str] = []
    xml_path = output_base_dir / "markers.xml"
    if xml_path.exists():
        tree = ET.parse(xml_path)
        cameras = sorted(tree.getroot().findall(".//camera"), key=lambda cam: cam.get("label", ""))

        for cam in cameras:
            transform = cam.find("transform")
            if transform is not None and transform.text is not None:
                label = cam.get("label")
                if label is not None:
                    valid_stems.append(label)

    # Map all images on disk by their filename stem
    image_map = {f.stem: f for f in image_dir.glob("*") if f.is_file()}

    # Rebuild the image list strictly matching the XML order and avoiding unaligned cameras
    if valid_stems:
        original_images = [image_map[stem] for stem in valid_stems if stem in image_map]
    else:
        # Fallback if XML is missing (shouldn't happen)
        original_images = sorted(list(image_map.values()), key=lambda x: x.name)

    if num_downscales <= 0:
        # No downscaling, return original filenames
        original_images = sorted([f for f in image_dir.glob("*") if f.is_file()])
        original_image_paths = [f"images/{img.name}" for img in original_images]

        original_dims = None
        if original_images:
            with Image.open(original_images[0]) as img:
                original_dims = img.size[::-1]  # PIL returns (width, height), we want (height, width)

        original_masks = None
        if mask_dir and mask_dir.exists():
            original_masks = sorted([f for f in mask_dir.glob("*") if f.is_file()])
            original_mask_paths: Optional[list[Optional[str]]] = [
                f"medium_masks/{mask.name}" for mask in original_masks
            ]
        else:
            original_mask_paths = None

        return (
            {1: original_image_paths},
            {1: original_mask_paths},
            {1: original_dims} if original_dims else {},
        )

    downscaled_image_filenames: dict[int, list[str]] = {}
    downscaled_mask_filenames: dict[int, Optional[list[Optional[str]]]] = {}
    image_dimensions: dict[int, tuple[int, int]] = {}

    # Get list of original images
    original_images = sorted([f for f in image_dir.glob("*") if f.is_file()])
    if not original_images:
        raise ValueError(f"No images found in {image_dir}")

    # Get list of original masks if available
    original_masks = None
    mask_map_by_stem: dict[str, Path] = {}
    if mask_dir and mask_dir.exists():
        original_masks = sorted([f for f in mask_dir.glob("*") if f.is_file()])
        for mask in original_masks:
            mask_map_by_stem[mask.stem] = mask

    if verbose:
        CONSOLE.log(f"[cyan]Downscaling {len(original_images)} images to {num_downscales} levels...[/cyan]")

    # Get original image dimensions
    with Image.open(original_images[0]) as img:
        orig_height, orig_width = img.size[1], img.size[0]  # PIL: (width, height)

    # Create downscaled versions
    for scale_level in range(1, num_downscales + 1):
        scale_factor = 2**scale_level  # 2x, 4x, 8x, etc.

        # Create output directories
        scaled_image_dir = output_base_dir / f"images_{scale_factor}"
        scaled_image_dir.mkdir(parents=True, exist_ok=True)

        scaled_mask_dir_out = None
        if original_masks:
            scaled_mask_dir_out = output_base_dir / f"medium_masks_{scale_factor}"
            scaled_mask_dir_out.mkdir(parents=True, exist_ok=True)

        scaled_image_paths: list[str] = []
        scaled_mask_paths: Optional[list[Optional[str]]] = [] if original_masks else None

        if verbose:
            CONSOLE.log(f"[cyan]  Creating {scale_factor}x downscaled images in {scaled_image_dir}...[/cyan]")

        new_height = orig_height // scale_factor
        new_width = orig_width // scale_factor

        for orig_image in original_images:
            # Open and downscale image
            with Image.open(orig_image) as img:
                scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save downscaled image
                output_image_path = scaled_image_dir / orig_image.name
                scaled_img.save(output_image_path, quality=95)
                scaled_image_paths.append(f"images_{scale_factor}/{orig_image.name}")

                # Downscale corresponding mask if it exists
                if original_masks and scaled_mask_dir_out and scaled_mask_paths is not None:
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
            CONSOLE.log(f"[cyan]    Created {len(scaled_image_paths)} images at {scale_factor}x downscale[/cyan]")
            CONSOLE.log(
                f"[cyan]    Created {len(scaled_mask_paths) if scaled_mask_paths is not None else 0} masks at {scale_factor}x downscale[/cyan]"
            )

    return downscaled_image_filenames, downscaled_mask_filenames, image_dimensions
