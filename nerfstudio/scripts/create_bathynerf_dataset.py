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

"""Create BathyNerfDataParser NPZ files from a Metashape XML export.

Thin orchestration script: parse XML → transform markers → fit water plane →
compute scene box → stratified train/val split → write NPZ files.

Helpers live in:
- nerfstudio/data/utils/metashape_xml_utils.py  (XML parsing, intrinsics, mask discovery, downscaling)
- nerfstudio/data/utils/bathy_normalization_utils.py  (coord transform, plane fit, R_norm selection, split)
"""

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.data.utils.bathy_normalization_utils import (
    NormalizationChain,
    calculate_scene_box,
    calculate_scene_box_from_mesh,
    fit_plane_to_points,
    select_normalization_rotation,
    stratified_train_val_split,
    transform_markers_to_chunk_local,
)
from nerfstudio.data.utils.metashape_xml_utils import (
    downscale_images_and_masks,
    extract_camera_intrinsics,
    extract_camera_transforms,
    extract_chunk_transform,
    extract_marker_coordinates,
    match_mask_filenames_to_images,
    scale_intrinsics,
)


def cleanup_existing_outputs(args: Any, output_dir: Path) -> None:
    """Remove existing output files and scaled directories that will be recreated in the current run."""
    print("\n--- Cleaning Up Existing Outputs ---")

    num_downscales = getattr(args, "num_downscales", getattr(args, "num_downscale", 0))
    has_masks = args.medium_mask_dir is not None

    # Clean up standard output files
    file_dict = {
        "train": output_dir / "train.npz",
        "val": output_dir / "val.npz",
        "data": output_dir / "data.npz",
        "visualization_left": output_dir / "visualization_left.png",
        "visualization_right": output_dir / "visualization_right.png",
        "visualization_top": output_dir / "visualization_top.png",
        "visualization_front": output_dir / "visualization_front.png",
    }

    for key, file_path in file_dict.items():
        if file_path.exists():
            print(f"Removing existing {key} file: {file_path}")
            file_path.unlink()

    # Clean up downscaled directories specifically for this run's parameters
    if num_downscales > 0:
        for scale_level in range(1, num_downscales + 1):
            scale_factor = 2**scale_level  # 2x, 4x, 8x, etc.

            # Check and delete scaled image directories
            scaled_image_dir = output_dir / f"images_{scale_factor}x"
            if scaled_image_dir.exists() and scaled_image_dir.is_dir():
                print(f"Removing existing scaled image directory: {scaled_image_dir}")
                shutil.rmtree(scaled_image_dir)

            # Check and delete scaled mask directories
            if has_masks:
                scaled_mask_dir = output_dir / f"medium_masks_{scale_factor}x"
                if scaled_mask_dir.exists() and scaled_mask_dir.is_dir():
                    print(f"Removing existing scaled mask directory: {scaled_mask_dir}")
                    shutil.rmtree(scaled_mask_dir)


def visualize_scene(
    camera_to_worlds: np.ndarray,
    markers: Optional[np.ndarray],
    water_normal: np.ndarray,
    water_d: float,
    scene_box: np.ndarray,
    output_dir: Path,
    intrinsics: dict[str, Any],
    aabb_scale: float,
    norm_scale: float = 1.0,
    chunk_scale: float = 1.0,
    extra_text: str = "",
) -> None:
    """Visualize the scene (cameras, markers, water plane, scene box) and save PNG views."""
    print("\n--- Generating Visualizations ---")

    # 1. Prepare Data
    cam_pos = camera_to_worlds[:, :3, 3]

    # Grid range covering scene box + margin
    # scene_box is typically (3, 2) => [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    if scene_box.shape == (3, 2):
        sb_min = scene_box[:, 0]
        sb_max = scene_box[:, 1]
    else:
        # Fallback if shape is (2, 3)
        sb_min = scene_box[0]
        sb_max = scene_box[1]

    margin = 0.5

    xx, yy = np.meshgrid(
        np.linspace(sb_min[0] - margin, sb_max[0] + margin, 10),
        np.linspace(sb_min[1] - margin, sb_max[1] + margin, 10),
    )

    n = water_normal
    if abs(n[2]) < 1e-3:
        zz = np.zeros_like(xx)  # Fallback
    else:
        zz = (-water_d - n[0] * xx - n[1] * yy) / n[2]

    # Metadata Text
    dist_params = intrinsics.get("distortion_params", [])
    k_params = f"k1={dist_params[0]:.4f}, k2={dist_params[1]:.4f}, k3={dist_params[2]:.4f}, k4={dist_params[3]:.4f}"
    p_params = f"p1={dist_params[4]:.4f}, p2={dist_params[5]:.4f}"

    marker_stats = "No Markers"
    if markers is not None and len(markers) > 0:
        msg_min = markers.min(axis=0)
        msg_max = markers.max(axis=0)
        marker_stats = (
            f"Markers: {len(markers)}\n"
            f"Min: [{msg_min[0]:.2f}, {msg_min[1]:.2f}, {msg_min[2]:.2f}]\n"
            f"Max: [{msg_max[0]:.2f}, {msg_max[1]:.2f}, {msg_max[2]:.2f}]"
        )

    # Camera Distances to Water Surface
    cam_dists_norm = np.dot(cam_pos, water_normal) + water_d
    cam_dists_metric = cam_dists_norm * norm_scale * chunk_scale

    dist_stats = (
        f"Cam-Water Dist (Metric):\n"
        f"  Min: {cam_dists_metric.min():.4f}\n"
        f"  Mean: {cam_dists_metric.mean():.4f}\n"
        f"  Max: {cam_dists_metric.max():.4f}"
    )

    info_text = (
        f"AABB Scale: {aabb_scale}\nDistortion:\n  {k_params}\n  {p_params}\n{marker_stats}\n{dist_stats}\n{extra_text}"
    )

    # 2. Plotting Function
    def save_view(elev: float, azim: float, name: str) -> None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2], c="blue", s=20, label="Cameras")

        if markers is not None and len(markers) > 0:
            ax.scatter(markers[:, 0], markers[:, 1], markers[:, 2], c="red", marker="^", s=50, label="Markers")

        ax.plot_surface(xx, yy, zz, alpha=0.3, color="cyan", label="Water Surface")

        # Draw edges of AABB
        for i in range(2):
            for j in range(2):
                # Z-lines
                ax.plot(
                    [sb_min[0] if i == 0 else sb_max[0]] * 2,
                    [sb_min[1] if j == 0 else sb_max[1]] * 2,
                    [sb_min[2], sb_max[2]],
                    "g--",
                    alpha=0.5,
                )
                # Y-lines
                ax.plot(
                    [sb_min[0] if i == 0 else sb_max[0]] * 2,
                    [sb_min[1], sb_max[1]],
                    [sb_min[2] if j == 0 else sb_max[2]] * 2,
                    "g--",
                    alpha=0.5,
                )
                # X-lines
                ax.plot(
                    [sb_min[0], sb_max[0]],
                    [sb_min[1] if i == 0 else sb_max[1]] * 2,
                    [sb_min[2] if j == 0 else sb_max[2]] * 2,
                    "g--",
                    alpha=0.5,
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Scene Visualization - {name}")
        ax.view_init(elev=elev, azim=azim)

        plt.figtext(0.02, 0.02, info_text, fontsize=9, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
        ax.legend()

        # Force equal aspect ratio hack
        max_range = np.array([sb_max[0] - sb_min[0], sb_max[1] - sb_min[1], sb_max[2] - sb_min[2]]).max() / 2.0
        mid_x = (sb_max[0] + sb_min[0]) * 0.5
        mid_y = (sb_max[1] + sb_min[1]) * 0.5
        mid_z = (sb_max[2] + sb_min[2]) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        out_path = output_dir / f"visualization_{name.lower()}.png"
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # 3. Generate Views
    save_view(elev=0, azim=0, name="Front")
    save_view(elev=90, azim=-90, name="Top")
    save_view(elev=0, azim=-90, name="Left")
    save_view(elev=0, azim=90, name="Right")


def main(args: Any) -> None:
    """Orchestrate Metashape XML → NPZ dataset creation pipeline."""
    output_dir = args.data_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect medium_masks if not provided
    if args.medium_mask_dir is None:
        potential_mask_dir = output_dir / "medium_masks"
        if potential_mask_dir.exists() and potential_mask_dir.is_dir():
            print(f"Auto-detected medium masks directory: {potential_mask_dir}")
            args.medium_mask_dir = potential_mask_dir

    xml_file = output_dir / args.xml_filename
    print(f"Reading data from {xml_file}...")

    # 0. Clean already existing npz files to avoid confusion
    cleanup_existing_outputs(args=args, output_dir=output_dir)

    # 1. Extract camera intrinsics (assuming they are the same for all cameras)
    intrinsics = extract_camera_intrinsics(xml_file)

    # 2. Extract camera-to-world matrices and generate image filenames
    camera_to_worlds, image_filenames = extract_camera_transforms(xml_file, image_extension=args.image_extension)
    camera_to_worlds = camera_to_worlds.astype(np.float32)

    num_cameras = len(camera_to_worlds)
    if num_cameras < 2:
        raise ValueError("At least two valid cameras are required to create a train/val split.")

    # 3. Extract and transform marker coordinates
    s_chunk, R_chunk, T_chunk = extract_chunk_transform(xml_file)
    markers_global = extract_marker_coordinates(xml_file)
    markers_local = transform_markers_to_chunk_local(markers_global, s_chunk, R_chunk, T_chunk).astype(np.float32)
    marker_labels = list(markers_global.keys())  # Preserve order for filtering

    # 4. Fit a plane to the markers (original, before normalization)
    original_water_normal, original_water_d, filter_mask, d_diff_to_unfiltered = fit_plane_to_points(
        markers_local, filter_outliers=True
    )
    original_water_normal = original_water_normal.astype(np.float32)
    original_water_d = np.float32(original_water_d)

    # Filter markers using the mask
    filtered_labels = [label for label, keep in zip(marker_labels, filter_mask) if keep]
    markers_global_filtered = {label: markers_global[label] for label in filtered_labels}
    markers_local_filtered = markers_local[filter_mask]

    print("\n--- Scene Normalization ---")
    print(f"Original water plane: normal={original_water_normal}, d={original_water_d:.4f}")
    print(
        f"Filter results: number of removed markers={np.sum(~filter_mask)}, d difference to unfiltered={d_diff_to_unfiltered:.6f}"
    )

    # 5. Compute normalization transforms (for reversibility)
    R_norm = select_normalization_rotation(R_chunk, original_water_normal)

    # Apply rotation to cameras
    rotated_camera_to_worlds = np.zeros_like(camera_to_worlds)
    for i in range(len(camera_to_worlds)):
        rotated_camera_to_worlds[i, :3, :3] = R_norm @ camera_to_worlds[i, :3, :3]
        rotated_camera_to_worlds[i, :3, 3] = R_norm @ camera_to_worlds[i, :3, 3]
        rotated_camera_to_worlds[i, 3, 3] = 1.0

    # Apply rotation to markers
    rotated_markers_filtered = (R_norm @ markers_local_filtered.T).T

    # Recompute water plane after rotation (should be [0, 0, 1] now)
    water_plane_normal, water_plane_d, _, _ = fit_plane_to_points(rotated_markers_filtered, filter_outliers=False)
    water_plane_normal = water_plane_normal.astype(np.float32)
    water_plane_d = np.float32(water_plane_d)
    print(f"After rotation: normal={water_plane_normal}, d={water_plane_d:.4f}")

    # 5b. Center: shift so that camera centroid is at origin (or water plane at z=0)
    camera_positions = rotated_camera_to_worlds[:, :3, 3]
    center = camera_positions.mean(axis=0).astype(np.float32)

    # Shift cameras
    centered_camera_to_worlds = rotated_camera_to_worlds.copy()
    centered_camera_to_worlds[:, :3, 3] -= center

    # Shift markers
    centered_markers_filtered = rotated_markers_filtered - center

    # Recompute water plane d after centering
    _, water_plane_d, filter_mask, _ = fit_plane_to_points(centered_markers_filtered, filter_outliers=False)
    water_plane_d = np.float32(water_plane_d)
    print(f"After centering: water plane d={water_plane_d:.4f}")

    # 5c. Scale: fit to approximate [-1, 1] range
    centered_positions = centered_camera_to_worlds[:, :3, 3]
    max_extent = np.max(np.abs(centered_positions))
    # Add a small margin to avoid clipping
    norm_scale = np.float32(max_extent * 1.1) if max_extent > 0 else np.float32(1.0)

    # Apply scale to cameras
    normalized_camera_to_worlds = centered_camera_to_worlds.copy()
    normalized_camera_to_worlds[:, :3, 3] /= norm_scale

    # Apply scale to markers
    normalized_markers_filtered = centered_markers_filtered / norm_scale

    # Scale water plane d
    water_plane_d = water_plane_d / norm_scale

    print(
        f"After scaling (factor={norm_scale:.4f}): positions in range [{normalized_camera_to_worlds[:, :3, 3].min():.3f}, {normalized_camera_to_worlds[:, :3, 3].max():.3f}]"
    )
    print(f"Final water plane: normal={water_plane_normal}, d={water_plane_d:.4f}")

    # Update camera_to_worlds and markers_local with normalized versions
    camera_to_worlds = normalized_camera_to_worlds.astype(np.float32)
    markers_local_filtered = normalized_markers_filtered.astype(np.float32)

    # 6. Create metadata for water surface (with normalized values)
    metadata: dict[str, Any] = {
        "water_surface": {
            "plane_model": {
                "normal": water_plane_normal.tolist(),
                "d": float(water_plane_d),
            },
            "source": "markers_from_minimal_dataparser",
        }
    }

    # Use nerfstudio's enum to ensure ids stay in sync with CameraType definitions
    camera_type = np.full(num_cameras, CameraType.PERSPECTIVE.value, dtype=np.int32)

    # 7. Calculate scene bounding box including markers
    if args.scene_box_from_mesh and args.underwater_scene:
        raise ValueError("--scene-box-from-mesh and --underwater-scene are mutually exclusive.")

    center_override = None
    aabb_center_used = args.aabb_center
    marker_centroid = None
    mesh_path_str: Optional[str] = None
    scene_box: Optional[np.ndarray] = None
    camera_centroid = camera_to_worlds[:, :3, 3].mean(axis=0)
    if markers_local_filtered is not None and len(markers_local_filtered) > 0:
        marker_centroid = markers_local_filtered.mean(axis=0)

    if args.scene_box_from_mesh:
        mesh_candidates = sorted(output_dir.glob("reference_mesh_*.ply"))
        if len(mesh_candidates) == 0:
            raise FileNotFoundError(f"--scene-box-from-mesh: no reference_mesh_*.ply found in {output_dir}")
        if len(mesh_candidates) > 1:
            raise ValueError(
                f"--scene-box-from-mesh: multiple reference_mesh_*.ply found in {output_dir}: "
                f"{[p.name for p in mesh_candidates]}. Resolve ambiguity before proceeding."
            )
        mesh_path_resolved = mesh_candidates[0]
        mesh_path_str = mesh_path_resolved.name
        scene_box = calculate_scene_box_from_mesh(
            mesh_path=mesh_path_resolved,
            water_plane_normal_scene=water_plane_normal,
            water_plane_d_scene=float(water_plane_d),
            s_chunk=float(s_chunk),
            R_chunk=R_chunk,
            T_chunk=T_chunk,
            R_norm=R_norm,
            center=center,
            norm_scale=float(norm_scale),
            buffer_m_global=float(args.scene_box_buffer_m),
        )
        aabb_center_used = "mesh_union_water"
        print(
            f"Scene box from mesh: {mesh_path_resolved.name} "
            f"(buffer={args.scene_box_buffer_m} m global); scene_box={scene_box.tolist()}"
        )
    elif args.underwater_scene:
        if marker_centroid is None:
            print("WARNING: --underwater-scene requested, but no markers found. Falling back to auto center.")
            aabb_center_used = "auto"
        else:
            all_points = camera_to_worlds[:, :3, 3]
            if markers_local_filtered is not None and len(markers_local_filtered) > 0:
                all_points = np.vstack((all_points, markers_local_filtered))
            min_bound = all_points.min(axis=0)
            max_bound = all_points.max(axis=0)
            auto_center = (min_bound + max_bound) / 2.0
            center_override = np.array(
                [camera_centroid[0], camera_centroid[1], auto_center[2]],
                dtype=np.float32,
            )
            aabb_center_used = "camera_xy_marker_top"
            print(f"Using camera-centered XY AABB for underwater scene (center={center_override})")
    elif args.aabb_center == "markers":
        if marker_centroid is None:
            print("WARNING: --aabb-center=markers requested, but no markers found. Falling back to auto center.")
            aabb_center_used = "auto"
        else:
            center_override = np.array(
                [camera_centroid[0], camera_centroid[1], marker_centroid[2]],
                dtype=np.float32,
            )
            aabb_center_used = "camera_xy_marker_center"
            print(f"Using camera-centered XY AABB with marker Z (center={center_override})")

    if not args.scene_box_from_mesh:
        scene_box = calculate_scene_box(
            camera_to_worlds,
            marker_positions=markers_local_filtered,
            aabb_scale=args.aabb_scale,
            center_override=center_override,
        ).astype(np.float32)
        if args.underwater_scene and marker_centroid is not None:
            z_shift = marker_centroid[2] - scene_box[2, 1]
            scene_box[2, :] += np.float32(z_shift)
            print(f"Shifted underwater scene box in Z by {z_shift:.6f} to align top with water surface.")
        print(f"Scene box computed with aabb_scale={args.aabb_scale}, center={args.aabb_center}")

    metadata["scene_box_config"] = {
        "aabb_scale": float(args.aabb_scale),
        "aabb_center": args.aabb_center,
        "aabb_center_used": aabb_center_used,
        "aabb_center_value": center_override.tolist() if center_override is not None else None,
        "underwater_scene": bool(args.underwater_scene),
        "scene_box_from_mesh": bool(args.scene_box_from_mesh),
        "scene_box_mesh_path": mesh_path_str,
        "scene_box_buffer_m": float(args.scene_box_buffer_m),
    }
    assert scene_box is not None  # exactly one of the branches above must have assigned it

    # Calculate Global Water Plane and Marker Heights
    ng = R_chunk @ original_water_normal
    dg = s_chunk * original_water_d - np.dot(ng, T_chunk)

    marker_heights_text = "Global Water Z per Marker:\n"
    if abs(ng[2]) > 1e-4:
        for lbl, p_g in markers_global_filtered.items():
            z_surface = -(dg + ng[0] * p_g[0] + ng[1] * p_g[1]) / ng[2]
            delta = p_g[2] - z_surface
            marker_heights_text += f"  {lbl}: Z={z_surface:.3f} (Δ={delta:.3f})\n"
    else:
        marker_heights_text += "  (Vertical water plane? Cannot compute Z)\n"

    # --- VISUALIZATION ---
    try:
        visualize_scene(
            camera_to_worlds,
            markers_local_filtered,
            water_plane_normal,
            water_plane_d,
            scene_box,
            output_dir,
            intrinsics,
            float(args.aabb_scale),
            float(norm_scale),
            float(s_chunk),
            extra_text=marker_heights_text,
        )
    except Exception as e:
        print(f"WARNING: Visualization failed: {e}")

    # 8. Split dataset into train/val with view-coverage across nadir/oblique and azimuth bins.
    train_indices, val_indices = stratified_train_val_split(camera_to_worlds, val_fraction=0.1)
    split_indices: dict[str, np.ndarray] = {
        "train": train_indices,
        "val": val_indices,
        "data": np.arange(num_cameras),
    }
    print(f"Train/val split: {len(train_indices)} train, {len(val_indices)} val (stratified coverage)")

    # Match medium masks to images if provided
    mask_filenames_array: Optional[np.ndarray] = None
    if args.medium_mask_dir is not None:
        mask_dir_resolved = (
            args.medium_mask_dir if args.medium_mask_dir.is_absolute() else output_dir / args.medium_mask_dir
        )
        print("\n--- Processing Medium Masks ---")
        print(f"Looking for masks in: {mask_dir_resolved}")
        matched_mask_paths = match_mask_filenames_to_images(image_filenames, mask_dir_resolved)

        # Validate that ALL images have matched masks (no None values allowed)
        num_matched = sum(1 for mask in matched_mask_paths if mask is not None)
        num_total = len(image_filenames)

        if num_matched == num_total:
            mask_filenames_array = np.array(matched_mask_paths)
            metadata["medium_mask_filenames"] = matched_mask_paths
            print(f"Added medium_mask_filenames to metadata ({num_matched}/{num_total} images)")
        elif num_matched > 0:
            print(f"WARNING: Only {num_matched}/{num_total} images have matching masks.")
            print("   All images must have masks. Skipping medium masks entirely.")
            print("   Missing masks for:")
            for i, (img, mask) in enumerate(zip(image_filenames, matched_mask_paths)):
                if mask is None:
                    print(f"     - {Path(img).name}")
                    if i >= 9:  # Show max 10 examples
                        remaining = num_total - num_matched - (i + 1)
                        if remaining > 0:
                            print(f"     ... and {remaining} more")
                        break
        else:
            print("WARNING: No masks were successfully matched. Proceeding without masks.")

    # 9. Perform downscaling if requested
    scale_factors = [1]  # Always include the base scale (1x = no downscaling)
    downscaled_image_filenames: dict[int, Any] = {1: image_filenames}
    downscaled_mask_filenames: dict[int, Any] = {
        1: None if mask_filenames_array is None else mask_filenames_array.tolist()
    }
    downscaled_intrinsics: dict[int, Any] = {1: intrinsics}

    if args.num_downscales > 0:
        print("\n--- Downscaling Images and Masks ---")
        images_dir = output_dir / "images"
        mask_dir_for_downscaling = None
        if args.medium_mask_dir is not None:
            resolved_path = (
                args.medium_mask_dir if args.medium_mask_dir.is_absolute() else output_dir / args.medium_mask_dir
            )
            if resolved_path.exists():
                mask_dir_for_downscaling = resolved_path

        downscaled_imgs, downscaled_masks, _ = downscale_images_and_masks(
            image_dir=images_dir,
            mask_dir=mask_dir_for_downscaling,
            output_base_dir=output_dir,
            num_downscales=args.num_downscales,
            verbose=True,
        )

        # Add downscaled versions to our tracking dicts
        for scale_factor, img_paths in downscaled_imgs.items():
            if scale_factor > 1:  # Skip the 1x (original) since we already have it
                scale_factors.append(scale_factor)
                downscaled_image_filenames[scale_factor] = img_paths
                downscaled_mask_filenames[scale_factor] = downscaled_masks.get(scale_factor)
                downscaled_intrinsics[scale_factor] = scale_intrinsics(intrinsics, scale_factor)

                scaled_int = downscaled_intrinsics[scale_factor]
                print(
                    f"  Scale {scale_factor}x: fx={scaled_int['fx']:.2f}, "
                    f"fy={scaled_int['fy']:.2f}, "
                    f"size={scaled_int['width']}x{scaled_int['height']}"
                )

    # Save NPZ files for each scale
    for scale_factor in scale_factors:
        scale_suffix = ""
        scale_dir = output_dir if scale_factor == 1 else output_dir / f"scale_{scale_factor}x"

        if scale_factor > 1:
            scale_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nProcessing {scale_factor}x downscaled dataset...")

        # Get image and mask filenames for this scale
        scale_image_filenames = np.array(downscaled_image_filenames[scale_factor])
        scale_mask_filenames_array = None
        if downscaled_mask_filenames[scale_factor] is not None:
            scale_mask_filenames_array = np.array(downscaled_mask_filenames[scale_factor])

        def slice_cameras(idx_array: np.ndarray, scale_factor: int = 1) -> dict[str, np.ndarray]:
            scaled_int = downscaled_intrinsics[scale_factor]
            return {
                "fx": np.full(len(idx_array), scaled_int["fx"], dtype=np.float32),
                "fy": np.full(len(idx_array), scaled_int["fy"], dtype=np.float32),
                "cx": np.full(len(idx_array), scaled_int["cx"], dtype=np.float32),
                "cy": np.full(len(idx_array), scaled_int["cy"], dtype=np.float32),
                "height": np.full(len(idx_array), scaled_int["height"], dtype=np.int32),
                "width": np.full(len(idx_array), scaled_int["width"], dtype=np.int32),
                "distortion_params": np.array([scaled_int["distortion_params"]] * len(idx_array), dtype=np.float32),
                "camera_to_worlds": camera_to_worlds[idx_array],
                "camera_type": camera_type[idx_array],
            }

        for split_name, idx_array in split_indices.items():
            split_output_path = scale_dir / f"{split_name}{scale_suffix}.npz"

            # Use deepcopy to avoid modifying the original nested dictionary (e.g., water_surface metadata)
            split_metadata: dict[str, Any] = copy.deepcopy(metadata)
            if scale_mask_filenames_array is not None:
                split_metadata["medium_mask_filenames"] = scale_mask_filenames_array[idx_array].tolist()

            cameras_blob: Any = slice_cameras(idx_array, scale_factor)
            metadata_blob: Any = split_metadata

            # Route the writer's normalization-chain assembly through the canonical
            # NormalizationChain dataclass. __post_init__ enforces R_norm := R_chunk
            # as a second gate alongside select_normalization_rotation (line ~292).
            # NPZ on-disk schema is byte-identical — fields are exploded back to the
            # same kwargs below.
            chain = NormalizationChain(
                norm_scale=float(norm_scale),
                norm_center=np.asarray(center, dtype=np.float32),
                norm_rot=np.asarray(R_norm, dtype=np.float32),
                chunk_scale=float(s_chunk),
                chunk_rot=R_chunk.astype(np.float32),
                chunk_trans=T_chunk.astype(np.float32),
            )

            np.savez(
                split_output_path,
                image_filenames=scale_image_filenames[idx_array],
                cameras=cameras_blob,
                scene_box=scene_box,
                marker_positions=markers_local_filtered,
                metadata=metadata_blob,
                # Original Metashape scale (global → chunk-local)
                applied_scale=np.float32(chain.chunk_scale),
                # Normalization parameters for reversibility (normalized → chunk-local)
                # To reverse: p_chunk_local = R_norm.T @ (p_normalized * norm_scale + center)
                # normalization_rotation == chunk_rotation for georeferenced datasets (R_chunk != I).
                normalization_rotation=chain.norm_rot,
                normalization_center=chain.norm_center,
                normalization_scale=np.float32(chain.norm_scale),
                original_water_normal=original_water_normal,
                original_water_d=original_water_d,
                # Metashape chunk transform: chunk-local → global
                # Verified 2026-05: chunk_{rotation,translation,scale} round-trip through
                # bathynerf_dataparser -> transform_points_to_bathy_global_frame at mm level
                # (tests/scripts/test_bathynerf_dataset_roundtrip.py).
                # To reverse to global: p_global = chunk_scale * (p_chunk_local @ chunk_rot.T) + chunk_trans
                chunk_rotation=chain.chunk_rot,
                chunk_translation=chain.chunk_trans,
            )
            print(f"Created: {split_output_path}")

    # Summary
    if args.num_downscales > 0:
        print("\nSuccessfully created NPZ files for multiple scales:")
        for scale_factor in scale_factors:
            scale_suffix_str = f"_{scale_factor}x" if scale_factor > 1 else ""
            print(
                f"  {scale_factor}x: train{scale_suffix_str}.npz, val{scale_suffix_str}.npz, data{scale_suffix_str}.npz"
            )
    else:
        print("\nSuccessfully created train.npz, val.npz, and data.npz for the BathyNerfDataParser.")
    print("\n--- Next Steps ---")

    print("1. Make sure your images are located in a folder named 'images' inside the data directory.")

    if args.medium_mask_dir is not None:
        print(f"2. Medium masks have been linked from: {args.medium_mask_dir}")
        print("   - Masks should have the same filename stems as images (e.g., 0001.png matches 0001.png)")
        print("   - Mask values: typically 0=land, 255=water, with optional intermediate class")
        print("   - The BathyNerfDataParser will load these masks automatically during training")
    else:
        print("2. OPTIONAL: To add medium masks, re-run this script with --medium-mask-dir:")
        print(f"   python {Path(__file__).name} --data-dir {output_dir} --medium-mask-dir medium_masks")

    if args.num_downscales > 0:
        print("\n2. OPTIONAL: To use downscaled datasets, specify a scale level in training:")
        print(f"   ns-train bathyfacto --data {output_dir}")
        print("   Or use a specific scale (e.g., 2x downscaled):")
        print(f"   ns-train bathyfacto --data {output_dir}/scale_2x")

    print("\n3. Run the nerfstudio training command:")
    print(f"   ns-train bathyfacto --data {output_dir}")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create BathyNerfDataParser npz files from markers.xml")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing markers.xml and where train/val/data npz files will be written.",
    )
    parser.add_argument(
        "--medium-mask-dir",
        type=Path,
        default=None,
        help="Directory containing medium mask images (e.g., 'medium_masks'). "
        "Mask filenames should match image filenames (e.g., 0001.png matches 0001.png). "
        "If not provided, masks will not be included in the NPZ files.",
    )
    parser.add_argument(
        "--aabb-scale",
        type=float,
        default=1.0,
        help="Scale factor for the scene AABB. 1.0 matches the current default behavior.",
    )
    parser.add_argument(
        "--aabb-center",
        type=str,
        choices=("auto", "markers"),
        default="auto",
        help="Center the AABB on the auto-computed bounds center or the marker centroid.",
    )
    parser.add_argument(
        "--underwater-scene",
        action="store_true",
        help=(
            "Center the AABB in XY on the camera centroid and shift it in Z so the"
            " top face aligns with the water surface (marker centroid Z)."
        ),
    )
    parser.add_argument(
        "--scene-box-from-mesh",
        action="store_true",
        help=(
            "Define the scene AABB from the GT mesh AABB unioned with the water surface,"
            " with an absolute global-metre buffer. Auto-discovers reference_mesh_*.ply in"
            " --data-dir. Mutually exclusive with --underwater-scene."
        ),
    )
    parser.add_argument(
        "--scene-box-buffer-m",
        type=float,
        default=0.5,
        help="Symmetric absolute buffer in global metres for --scene-box-from-mesh (default: 0.5).",
    )
    parser.add_argument(
        "--xml-filename",
        type=str,
        default="markers.xml",
        help="Name of the XML file to process within the data directory.",
    )
    parser.add_argument(
        "--image-extension",
        type=str,
        default=".png",
        choices=[".png", ".jpg", ".JPG", ".jpeg"],
        help="File extension for images (default: .png). Supports .png, .jpg, .JPG, .jpeg.",
    )
    parser.add_argument(
        "--num-downscales",
        type=int,
        default=0,
        help="Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 "
        "will create downscaled images at 2x, 4x, and 8x. Default is 0 (no downscaling).",
    )
    args = parser.parse_args()

    main(args)
