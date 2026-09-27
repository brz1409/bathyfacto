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

"""Create BathyNerfDataParser NPZ files from a Metashape XML export.

Thin orchestration script: parse XML → transform markers → fit water plane →
compute scene box → stratified train/val split → write NPZ files.

Helpers live in:
- bathyfacto/metashape_xml.py  (XML parsing, intrinsics, mask discovery, downscaling)
- bathyfacto/normalization.py  (coord transform, plane fit, R_norm selection, split)
"""

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.utils.rich_utils import CONSOLE

from bathyfacto.metashape_xml import (
    downscale_images_and_masks,
    extract_camera_intrinsics,
    extract_camera_transforms,
    extract_chunk_transform,
    extract_marker_coordinates,
    match_mask_filenames_to_images,
    parse_metashape_xml,
    scale_intrinsics,
)
from bathyfacto.normalization import (
    NormalizationChain,
    calculate_scene_box,
    calculate_scene_box_from_mesh,
    fit_plane_to_points,
    select_normalization_rotation,
    stratified_train_val_split,
    transform_markers_to_chunk_local,
)


def cleanup_existing_outputs(args: Any, output_dir: Path) -> None:
    """Remove existing output files and scaled directories that will be recreated in the current run."""
    num_downscales = args.num_downscales
    has_masks = args.medium_mask_dir is not None
    removed: list[str] = []

    file_dict = {
        "train": output_dir / "train.npz",
        "val": output_dir / "val.npz",
        "data": output_dir / "data.npz",
        "visualization_left": output_dir / "visualization_left.png",
        "visualization_right": output_dir / "visualization_right.png",
        "visualization_top": output_dir / "visualization_top.png",
        "visualization_front": output_dir / "visualization_front.png",
    }

    for file_path in file_dict.values():
        if file_path.exists():
            file_path.unlink()
            removed.append(file_path.name)

    if num_downscales > 0:
        for scale_level in range(1, num_downscales + 1):
            scale_factor = 2**scale_level  # 2x, 4x, 8x, etc.

            # downscale_images_and_masks (in bathyfacto.metashape_xml) writes "images_{scale_factor}", with no "x" suffix;
            # the suffix belongs to the downscaled NPZ names (train_2x.npz), not the folders.
            scaled_image_dir = output_dir / f"images_{scale_factor}"
            if scaled_image_dir.exists() and scaled_image_dir.is_dir():
                shutil.rmtree(scaled_image_dir)
                removed.append(scaled_image_dir.name)

            if has_masks:
                scaled_mask_dir = output_dir / f"medium_masks_{scale_factor}"
                if scaled_mask_dir.exists() and scaled_mask_dir.is_dir():
                    shutil.rmtree(scaled_mask_dir)
                    removed.append(scaled_mask_dir.name)

    if removed:
        CONSOLE.print(
            f"bathyfacto  dataset: removed {len(removed)} existing outputs in {output_dir}: {', '.join(removed)}"
        )
    else:
        CONSOLE.print(f"bathyfacto  dataset: no existing outputs to remove in {output_dir}")


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

    cam_pos = camera_to_worlds[:, :3, 3]

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

    dist_params = intrinsics["distortion_params"]
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

    def save_view(elev: float, azim: float, name: str) -> None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # matplotlib's Axes3D.scatter stub types zs as a plain int; the real signature also
        # accepts an array of per-point z-values, which is what we pass here.
        ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cast(int, cam_pos[:, 2]), c="blue", s=20, label="Cameras")

        if markers is not None and len(markers) > 0:
            ax.scatter(
                markers[:, 0], markers[:, 1], cast(int, markers[:, 2]), c="red", marker="^", s=50, label="Markers"
            )

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

    save_view(elev=0, azim=0, name="Front")
    save_view(elev=90, azim=-90, name="Top")
    save_view(elev=0, azim=-90, name="Left")
    save_view(elev=0, azim=90, name="Right")
    CONSOLE.print(f"bathyfacto  dataset: 4 visualization renders written to {output_dir}")


def _normalize_scene(
    camera_to_worlds: np.ndarray,
    markers_global: dict[str, np.ndarray],
    s_chunk: float,
    R_chunk: np.ndarray,
    T_chunk: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.float32, np.ndarray, np.float32, NormalizationChain
]:
    """Rotate, center and scale cameras and markers into the scene frame; fit the water plane.

    Returns the scene-frame cameras and inlier markers, the inlier markers in the global frame,
    the scene-frame water plane (normal, d), the chunk-local water plane (normal, d) and the chain.
    """
    markers_local = transform_markers_to_chunk_local(markers_global, s_chunk, R_chunk, T_chunk).astype(np.float32)

    # Water plane from the markers, chunk-local, before normalization.
    original_water_normal, original_water_d, filter_mask = fit_plane_to_points(markers_local, filter_outliers=True)
    original_water_normal = original_water_normal.astype(np.float32)
    original_water_d = np.float32(original_water_d)
    num_markers_removed = int(np.sum(~filter_mask))

    markers_global_filtered = {
        label: markers_global[label] for label, keep in zip(markers_global.keys(), filter_mask) if keep
    }
    markers_local_filtered = markers_local[filter_mask]

    R_norm = select_normalization_rotation(R_chunk, original_water_normal)

    rotated_camera_to_worlds = np.zeros_like(camera_to_worlds)
    for i in range(len(camera_to_worlds)):
        rotated_camera_to_worlds[i, :3, :3] = R_norm @ camera_to_worlds[i, :3, :3]
        rotated_camera_to_worlds[i, :3, 3] = R_norm @ camera_to_worlds[i, :3, 3]
        rotated_camera_to_worlds[i, 3, 3] = 1.0

    rotated_markers_filtered = (R_norm @ markers_local_filtered.T).T

    # Refit the plane in the rotated frame. Its normal is [0, 0, 1] only on the
    # align_water_to_horizontal fallback (no chunk transform).
    water_plane_normal, _, _ = fit_plane_to_points(rotated_markers_filtered, filter_outliers=False)
    water_plane_normal = water_plane_normal.astype(np.float32)

    # Center: the camera centroid moves to the origin.
    center = rotated_camera_to_worlds[:, :3, 3].mean(axis=0).astype(np.float32)

    centered_camera_to_worlds = rotated_camera_to_worlds.copy()
    centered_camera_to_worlds[:, :3, 3] -= center

    centered_markers_filtered = rotated_markers_filtered - center

    _, water_plane_d, _ = fit_plane_to_points(centered_markers_filtered, filter_outliers=False)
    water_plane_d = np.float32(water_plane_d)

    # Scale the camera positions into roughly [-1, 1].
    max_extent = np.max(np.abs(centered_camera_to_worlds[:, :3, 3]))
    # Add a small margin to avoid clipping
    norm_scale = np.float32(max_extent * 1.1) if max_extent > 0 else np.float32(1.0)

    normalized_camera_to_worlds = centered_camera_to_worlds.copy()
    normalized_camera_to_worlds[:, :3, 3] /= norm_scale

    water_plane_d = water_plane_d / norm_scale

    CONSOLE.print(
        f"bathyfacto  dataset: scene normalized, {num_markers_removed} outlier markers filtered, "
        f"scale factor {float(norm_scale):.4f}, water plane normal {water_plane_normal.tolist()} "
        f"d={float(water_plane_d):.4f}"
    )

    # Checks R_norm := R_chunk before any value is written.
    chain = NormalizationChain(
        norm_scale=float(norm_scale),
        norm_center=np.asarray(center, dtype=np.float32),
        norm_rot=np.asarray(R_norm, dtype=np.float32),
        chunk_scale=float(s_chunk),
        chunk_rot=R_chunk.astype(np.float32),
        chunk_trans=T_chunk.astype(np.float32),
    )
    return (
        normalized_camera_to_worlds.astype(np.float32),
        (centered_markers_filtered / norm_scale).astype(np.float32),
        markers_global_filtered,
        water_plane_normal,
        water_plane_d,
        original_water_normal,
        original_water_d,
        chain,
    )


def _compute_scene_box(
    args: Any,
    output_dir: Path,
    camera_to_worlds: np.ndarray,
    markers: np.ndarray,
    water_plane_normal: np.ndarray,
    water_plane_d: np.float32,
    chain: NormalizationChain,
    R_chunk: np.ndarray,
    T_chunk: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scene box (3, 2) in the scene frame and its ``scene_box_config`` metadata."""
    if args.scene_box_from_mesh and args.underwater_scene:
        raise ValueError("--scene-box-from-mesh and --underwater-scene are mutually exclusive.")

    center_override = None
    aabb_center_used = args.aabb_center
    mesh_path_str: Optional[str] = None
    camera_centroid = camera_to_worlds[:, :3, 3].mean(axis=0)
    marker_centroid = markers.mean(axis=0) if len(markers) > 0 else None

    if args.scene_box_from_mesh:
        mesh_candidates = sorted(output_dir.glob("reference_mesh_*.ply"))
        if len(mesh_candidates) == 0:
            raise FileNotFoundError(f"--scene-box-from-mesh: no reference_mesh_*.ply found in {output_dir}")
        if len(mesh_candidates) > 1:
            raise ValueError(
                f"--scene-box-from-mesh: multiple reference_mesh_*.ply found in {output_dir}: "
                f"{[p.name for p in mesh_candidates]}. Resolve ambiguity before proceeding."
            )
        mesh_path_str = mesh_candidates[0].name
        # Logs the one summary line for this event (mesh name, vertex count, buffer, box min/max).
        scene_box = calculate_scene_box_from_mesh(
            mesh_path=mesh_candidates[0],
            water_plane_normal_scene=water_plane_normal,
            water_plane_d_scene=float(water_plane_d),
            s_chunk=chain.chunk_scale,
            R_chunk=R_chunk,
            T_chunk=T_chunk,
            R_norm=chain.norm_rot,
            center=chain.norm_center,
            norm_scale=chain.norm_scale,
            buffer_m_global=float(args.scene_box_buffer_m),
        )
        aabb_center_used = "mesh_union_water"
    else:
        if args.underwater_scene:
            if marker_centroid is None:
                CONSOLE.log(
                    "[yellow]bathyfacto  dataset: --underwater-scene has no markers, using auto center[/yellow]"
                )
                aabb_center_used = "auto"
            else:
                all_points = np.vstack((camera_to_worlds[:, :3, 3], markers))
                auto_center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2.0
                center_override = np.array(
                    [camera_centroid[0], camera_centroid[1], auto_center[2]],
                    dtype=np.float32,
                )
                aabb_center_used = "camera_xy_marker_top"
                CONSOLE.print(f"bathyfacto  dataset: underwater scene AABB centered at {center_override}")
        elif args.aabb_center == "markers":
            if marker_centroid is None:
                CONSOLE.log(
                    "[yellow]bathyfacto  dataset: --aabb-center=markers has no markers, using auto center[/yellow]"
                )
                aabb_center_used = "auto"
            else:
                center_override = np.array(
                    [camera_centroid[0], camera_centroid[1], marker_centroid[2]],
                    dtype=np.float32,
                )
                aabb_center_used = "camera_xy_marker_center"
                CONSOLE.print(f"bathyfacto  dataset: AABB centered on markers at {center_override}")

        scene_box = calculate_scene_box(
            camera_to_worlds,
            marker_positions=markers,
            aabb_scale=args.aabb_scale,
            center_override=center_override,
        ).astype(np.float32)
        z_shift_note = ""
        if args.underwater_scene and marker_centroid is not None:
            z_shift = marker_centroid[2] - scene_box[2, 1]
            scene_box[2, :] += np.float32(z_shift)
            z_shift_note = f", shifted in Z by {z_shift:.6f} to align the top with the water surface"
        CONSOLE.print(
            f"bathyfacto  dataset: scene box computed with aabb_scale={args.aabb_scale}, "
            f"center={args.aabb_center}{z_shift_note}"
        )

    scene_box_config = {
        "aabb_scale": float(args.aabb_scale),
        "aabb_center": args.aabb_center,
        "aabb_center_used": aabb_center_used,
        "aabb_center_value": center_override.tolist() if center_override is not None else None,
        "underwater_scene": bool(args.underwater_scene),
        "scene_box_from_mesh": bool(args.scene_box_from_mesh),
        "scene_box_mesh_path": mesh_path_str,
        "scene_box_buffer_m": float(args.scene_box_buffer_m),
    }
    return scene_box, scene_box_config


def _marker_heights_text(
    markers_global: dict[str, np.ndarray],
    s_chunk: float,
    R_chunk: np.ndarray,
    T_chunk: np.ndarray,
    original_water_normal: np.ndarray,
    original_water_d: np.float32,
) -> str:
    """Global water height below each marker, for the visualization caption."""
    ng = R_chunk @ original_water_normal
    dg = s_chunk * original_water_d - np.dot(ng, T_chunk)

    text = "Global Water Z per Marker:\n"
    if abs(ng[2]) > 1e-4:
        for lbl, p_g in markers_global.items():
            z_surface = -(dg + ng[0] * p_g[0] + ng[1] * p_g[1]) / ng[2]
            delta = p_g[2] - z_surface
            text += f"  {lbl}: Z={z_surface:.3f} (Δ={delta:.3f})\n"
    else:
        text += "  (Vertical water plane? Cannot compute Z)\n"
    return text


def _match_medium_masks(image_filenames: list[str], mask_dir: Path) -> Optional[list[Optional[str]]]:
    """Mask path per image, or ``None`` unless every image has one."""
    matched_mask_paths = match_mask_filenames_to_images(image_filenames, mask_dir)

    num_matched = sum(1 for mask in matched_mask_paths if mask is not None)
    num_total = len(image_filenames)

    if num_matched == num_total:
        CONSOLE.print(f"bathyfacto  dataset: medium_mask_filenames added to metadata for {num_matched} images")
        return matched_mask_paths
    if num_matched > 0:
        missing = [Path(img).name for img, mask in zip(image_filenames, matched_mask_paths) if mask is None]
        shown = ", ".join(missing[:10])
        if len(missing) > 10:
            shown += f", and {len(missing) - 10} more"
        CONSOLE.log(
            f"[yellow]bathyfacto  dataset: only {num_matched}/{num_total} images have matching masks, "
            f"skipping medium masks entirely, missing masks for {shown}[/yellow]"
        )
    else:
        CONSOLE.log("[yellow]bathyfacto  dataset: no masks matched, proceeding without medium masks[/yellow]")
    return None


def _slice_cameras(
    intrinsics: dict[str, Any], camera_to_worlds: np.ndarray, camera_type: np.ndarray, idx_array: np.ndarray
) -> dict[str, np.ndarray]:
    n = len(idx_array)
    return {
        "fx": np.full(n, intrinsics["fx"], dtype=np.float32),
        "fy": np.full(n, intrinsics["fy"], dtype=np.float32),
        "cx": np.full(n, intrinsics["cx"], dtype=np.float32),
        "cy": np.full(n, intrinsics["cy"], dtype=np.float32),
        "height": np.full(n, intrinsics["height"], dtype=np.int32),
        "width": np.full(n, intrinsics["width"], dtype=np.int32),
        "distortion_params": np.array([intrinsics["distortion_params"]] * n, dtype=np.float32),
        "camera_to_worlds": camera_to_worlds[idx_array],
        "camera_type": camera_type[idx_array],
    }


def _write_splits(
    output_dir: Path,
    levels: dict[int, tuple[Any, Optional[list[Optional[str]]], dict[str, Any]]],
    split_indices: dict[str, np.ndarray],
    camera_to_worlds: np.ndarray,
    scene_box: np.ndarray,
    markers: np.ndarray,
    metadata: dict[str, Any],
    chain: NormalizationChain,
    original_water_normal: np.ndarray,
    original_water_d: np.float32,
) -> list[tuple[str, int]]:
    """Write one ``<split>[_<factor>x].npz`` per scale level and split; return (name, size) pairs."""
    # Use nerfstudio's enum so ids stay in sync with CameraType definitions
    camera_type = np.full(len(camera_to_worlds), CameraType.PERSPECTIVE.value, dtype=np.int32)
    written_splits: list[tuple[str, int]] = []
    for scale_factor, (image_filenames, mask_filenames, intrinsics) in levels.items():
        # Downscaled splits are written into the dataset root with an `_<factor>x` suffix
        # (train_2x.npz). The summary below prints these names and
        # BathyNerfDataParser._split_candidates looks for them.
        scale_suffix = "" if scale_factor == 1 else f"_{scale_factor}x"
        scale_image_filenames = np.array(image_filenames)
        scale_mask_filenames_array = None if mask_filenames is None else np.array(mask_filenames)

        for split_name, idx_array in split_indices.items():
            split_output_path = output_dir / f"{split_name}{scale_suffix}.npz"

            # Use deepcopy to avoid modifying the original nested dictionary (e.g., water_surface metadata)
            split_metadata: dict[str, Any] = copy.deepcopy(metadata)
            if scale_mask_filenames_array is not None:
                split_metadata["medium_mask_filenames"] = scale_mask_filenames_array[idx_array].tolist()

            np.savez(
                split_output_path,
                image_filenames=scale_image_filenames[idx_array],
                cameras=cast(Any, _slice_cameras(intrinsics, camera_to_worlds, camera_type, idx_array)),
                scene_box=scene_box,
                marker_positions=markers,
                metadata=cast(Any, split_metadata),
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
                # To reverse to global: p_global = chunk_scale * (p_chunk_local @ chunk_rot.T) + chunk_trans
                chunk_rotation=chain.chunk_rot,
                chunk_translation=chain.chunk_trans,
            )
            written_splits.append((split_output_path.name, len(idx_array)))
    return written_splits


def main(args: Any) -> None:
    """Orchestrate Metashape XML → NPZ dataset creation pipeline."""
    output_dir = args.data_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.medium_mask_dir is None:
        potential_mask_dir = output_dir / "medium_masks"
        if potential_mask_dir.exists() and potential_mask_dir.is_dir():
            CONSOLE.print(f"bathyfacto  dataset: auto-detected medium masks directory {potential_mask_dir}")
            args.medium_mask_dir = potential_mask_dir

    xml_file = output_dir / args.xml_filename
    CONSOLE.print(f"bathyfacto  dataset: reading {xml_file}")

    # Remove the outputs of an earlier run so stale splits cannot mix with new ones.
    cleanup_existing_outputs(args=args, output_dir=output_dir)

    xml_root = parse_metashape_xml(xml_file)
    # All cameras share one sensor calibration.
    intrinsics = extract_camera_intrinsics(xml_root)

    camera_to_worlds, image_filenames = extract_camera_transforms(xml_root, image_extension=args.image_extension)
    camera_to_worlds = camera_to_worlds.astype(np.float32)

    num_cameras = len(camera_to_worlds)
    if num_cameras < 2:
        raise ValueError("At least two valid cameras are required to create a train/val split.")

    s_chunk, R_chunk, T_chunk = extract_chunk_transform(xml_root)
    markers_global = extract_marker_coordinates(xml_root)
    (
        camera_to_worlds,
        markers,
        markers_global_filtered,
        water_plane_normal,
        water_plane_d,
        original_water_normal,
        original_water_d,
        chain,
    ) = _normalize_scene(camera_to_worlds, markers_global, s_chunk, R_chunk, T_chunk)

    metadata: dict[str, Any] = {
        "water_surface": {
            "plane_model": {
                "normal": water_plane_normal.tolist(),
                "d": float(water_plane_d),
            },
            "source": "markers_from_minimal_dataparser",
        }
    }

    scene_box, metadata["scene_box_config"] = _compute_scene_box(
        args, output_dir, camera_to_worlds, markers, water_plane_normal, water_plane_d, chain, R_chunk, T_chunk
    )

    try:
        visualize_scene(
            camera_to_worlds,
            markers,
            water_plane_normal,
            float(water_plane_d),
            scene_box,
            output_dir,
            intrinsics,
            float(args.aabb_scale),
            chain.norm_scale,
            chain.chunk_scale,
            extra_text=_marker_heights_text(
                markers_global_filtered, s_chunk, R_chunk, T_chunk, original_water_normal, original_water_d
            ),
        )
    except Exception as e:
        CONSOLE.log(f"[yellow]bathyfacto  dataset: visualization failed, {e}[/yellow]")

    # Split into train/val with view-coverage across nadir/oblique and azimuth bins.
    train_indices, val_indices = stratified_train_val_split(camera_to_worlds, val_fraction=0.1)
    split_indices: dict[str, np.ndarray] = {
        "train": train_indices,
        "val": val_indices,
        "data": np.arange(num_cameras),
    }
    CONSOLE.print(
        f"bathyfacto  dataset: train/val split, {len(train_indices)} train, {len(val_indices)} val, stratified coverage"
    )

    mask_dir: Optional[Path] = None
    if args.medium_mask_dir is not None:
        given_mask_dir: Path = args.medium_mask_dir
        mask_dir = given_mask_dir if given_mask_dir.is_absolute() else output_dir / given_mask_dir
    mask_filenames = None if mask_dir is None else _match_medium_masks(image_filenames, mask_dir)
    if mask_filenames is not None:
        metadata["medium_mask_filenames"] = mask_filenames

    # Scale level -> (image filenames, medium mask filenames or None, intrinsics). 1x is the original.
    levels: dict[int, tuple[Any, Optional[list[Optional[str]]], dict[str, Any]]] = {
        1: (image_filenames, mask_filenames, intrinsics)
    }
    if args.num_downscales > 0:
        # A given mask directory exists here: matching the masks above raises otherwise.
        downscaled_imgs, downscaled_masks, _ = downscale_images_and_masks(
            image_dir=output_dir / "images",
            mask_dir=mask_dir,
            output_base_dir=output_dir,
            num_downscales=args.num_downscales,
            verbose=True,
            xml_root=xml_root,
        )

        scaled_intrinsics_summary: list[str] = []
        for scale_factor, img_paths in downscaled_imgs.items():
            scaled_int = scale_intrinsics(intrinsics, scale_factor)
            levels[scale_factor] = (img_paths, downscaled_masks.get(scale_factor), scaled_int)
            scaled_intrinsics_summary.append(
                f"{scale_factor}x fx={scaled_int['fx']:.2f} fy={scaled_int['fy']:.2f} "
                f"size={scaled_int['width']}x{scaled_int['height']}"
            )

        if scaled_intrinsics_summary:
            CONSOLE.print("bathyfacto  dataset: downscaled intrinsics, " + ", ".join(scaled_intrinsics_summary))

    written_splits = _write_splits(
        output_dir,
        levels,
        split_indices,
        camera_to_worlds,
        scene_box,
        markers,
        metadata,
        chain,
        original_water_normal,
        original_water_d,
    )

    split_sizes = ", ".join(f"{name} ({count} images)" for name, count in written_splits)
    CONSOLE.print(f"bathyfacto  dataset: {len(written_splits)} split files written to {output_dir}, {split_sizes}")

    CONSOLE.print("bathyfacto  dataset: images must live in an 'images' folder inside the data directory")

    if args.medium_mask_dir is not None:
        CONSOLE.print(
            f"bathyfacto  dataset: medium masks linked from {args.medium_mask_dir}, matched by filename stem, "
            "values 0=land 255=water with an optional intermediate class, "
            "loaded automatically by BathyNerfDataParser"
        )
    else:
        CONSOLE.print(
            "bathyfacto  dataset: no medium masks, add them by re-running with --medium-mask-dir, "
            f"for example bathyfacto-build-dataset --data-dir {output_dir} --medium-mask-dir medium_masks"
        )

    if args.num_downscales > 0:
        downscale_factors = [s for s in levels if s > 1]
        CONSOLE.print(
            f"bathyfacto  dataset: downscaled splits available at factors {downscale_factors}, select one with "
            "ns-train bathyfacto --data <data-dir> bathynerf --downscale-factor 2 (for example)"
        )

    CONSOLE.print(f"bathyfacto  dataset: to train, run ns-train bathyfacto --data {output_dir}")


def entrypoint() -> None:
    parser = argparse.ArgumentParser(description="Create BathyNerfDataParser npz files from markers.xml")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the Metashape XML (see --xml-filename) and the images folder. "
        "The train/val/data npz files are written here.",
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
    main(parser.parse_args())


if __name__ == "__main__":
    entrypoint()
