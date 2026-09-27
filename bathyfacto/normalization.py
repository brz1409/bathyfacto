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

"""Scene normalization for the dataset builder.

Chunk-local transform, water-plane fit, R_norm := R_chunk selection, horizontal-alignment
fallback, stratified train/val split.

For georeferenced datasets (R_chunk != I), normalization_rotation == chunk_rotation, so the
scene frame equals the true world frame up to pure translation and uniform scale. Points are
mapped between the frames by ``NormalizationChain``:

    chunk_local = (normalized * norm_scale + norm_center) @ norm_rot
    global      = chunk_scale * (chunk_local @ chunk_rot.T) + chunk_trans   # scene_to_global
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, Union

import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE

Points = TypeVar("Points", np.ndarray, torch.Tensor)


@dataclass(frozen=True)
class NormalizationChain:
    """Frozen runtime view over the scene↔global normalization chain stored in NPZ files.

    Fields mirror the NPZ keys 1:1. :meth:`scene_to_global` and :meth:`global_to_scene` are the
    one place the chain is applied to points:

        scene → chunk_local = (pts * norm_scale + norm_center) @ norm_rot
              → global      = chunk_scale * (chunk_local @ chunk_rot.T) + chunk_trans

    Invariant (georeferenced datasets): ``norm_rot == chunk_rot`` (the
    ``R_norm := R_chunk`` rule, see ``select_normalization_rotation``). ``__post_init__``
    enforces this within ``atol=1e-6`` whenever ``chunk_rot`` is non-identity. When
    ``chunk_rot ≈ I`` (the non-georeferenced fallback used by
    ``align_water_to_horizontal``), any ``norm_rot`` is allowed.
    """

    norm_scale: float
    norm_center: np.ndarray  # shape (3,)
    norm_rot: np.ndarray  # shape (3, 3)
    chunk_scale: float
    chunk_rot: np.ndarray  # shape (3, 3)
    chunk_trans: np.ndarray  # shape (3,)

    def __post_init__(self) -> None:
        # Shape checks: fail fast on wrong inputs.
        if self.norm_center.shape != (3,):
            raise ValueError(f"norm_center must have shape (3,), got {self.norm_center.shape}")
        if self.norm_rot.shape != (3, 3):
            raise ValueError(f"norm_rot must have shape (3, 3), got {self.norm_rot.shape}")
        if self.chunk_rot.shape != (3, 3):
            raise ValueError(f"chunk_rot must have shape (3, 3), got {self.chunk_rot.shape}")
        if self.chunk_trans.shape != (3,):
            raise ValueError(f"chunk_trans must have shape (3,), got {self.chunk_trans.shape}")
        # R_norm := R_chunk invariant (georeferenced datasets only, the identity fallback is allowed).
        chunk_is_identity = np.allclose(self.chunk_rot, np.eye(3), atol=1e-6)
        if not chunk_is_identity and not np.allclose(self.norm_rot, self.chunk_rot, atol=1e-6):
            max_diff = float(np.max(np.abs(self.norm_rot - self.chunk_rot)))
            raise ValueError(
                "NormalizationChain invariant violated: norm_rot != chunk_rot "
                f"(max |diff| = {max_diff:.3e}, atol=1e-6). For georeferenced datasets "
                "the dataset builder must set R_norm := R_chunk (see "
                "`select_normalization_rotation` in this module)."
            )

    def _like(self, value: Any, points: Union[np.ndarray, torch.Tensor]) -> Any:
        """``value`` as a tensor matching ``points``, or unchanged for numpy points."""
        if isinstance(points, torch.Tensor):
            return torch.as_tensor(value, dtype=points.dtype, device=points.device)
        return value

    def scene_to_global(self, points: Points) -> Points:
        """Map ``(N, 3)`` scene points to the global frame (numpy or torch, dtype follows ``points``)."""
        norm_rot = self._like(self.norm_rot, points)
        norm_center = self._like(self.norm_center, points)
        chunk_rot = self._like(self.chunk_rot, points)
        chunk_trans = self._like(self.chunk_trans, points)

        chunk_local_points = (points * self.norm_scale + norm_center) @ norm_rot
        return self.chunk_scale * (chunk_local_points @ chunk_rot.T) + chunk_trans

    def global_to_scene(self, points: np.ndarray) -> np.ndarray:
        """Map ``(N, 3)`` global points to the scene frame.

        Chain: chunk_local = (1/s_chunk) * R_chunk.T @ (p_global - T_chunk);
        rotated = R_norm @ chunk_local; centered = rotated - center; scene = centered / norm_scale.
        """
        chunk_local = ((points - self.chunk_trans) @ self.chunk_rot) / self.chunk_scale
        rotated = chunk_local @ self.norm_rot.T
        centered = rotated - self.norm_center[None, :]
        return centered / self.norm_scale

    @classmethod
    def from_npz_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        dataparser_scale: float | None = None,
    ) -> "NormalizationChain":
        """Build a chain from a metadata-style mapping.

        Accepts both the long key names used by the dataparser at runtime
        (``normalization_rotation``, ``normalization_center``, ``normalization_scale``,
        ``chunk_rotation``, ``chunk_translation``) and the short field names of this
        dataclass (``norm_rot``, ``norm_center``, ``norm_scale``, ``chunk_rot``,
        ``chunk_trans``). When ``dataparser_scale`` is given it overrides any
        ``chunk_scale`` value in the mapping (this matches
        ``transform_points_to_bathy_global_frame``, which reads ``chunk_scale`` from
        ``dataparser_outputs.dataparser_scale`` and not from the metadata dict).

        Args:
            metadata: Mapping carrying the six chain quantities under either naming.
            dataparser_scale: Optional override for ``chunk_scale`` (typically
                ``DataparserOutputs.dataparser_scale``).

        Returns:
            A frozen ``NormalizationChain``.

        Raises:
            KeyError: If a required chain quantity is absent from ``metadata`` (and not
                supplied via ``dataparser_scale`` for the chunk scale).
            ValueError: If the R_norm := R_chunk invariant is violated.
        """

        def _pick(*keys: str) -> Any:
            for key in keys:
                if key in metadata:
                    return metadata[key]
            return None

        norm_rot_val = _pick("normalization_rotation", "norm_rot")
        norm_center_val = _pick("normalization_center", "norm_center")
        norm_scale_val = _pick("normalization_scale", "norm_scale")
        chunk_rot_val = _pick("chunk_rotation", "chunk_rot")
        chunk_trans_val = _pick("chunk_translation", "chunk_trans")
        chunk_scale_val: Any = dataparser_scale if dataparser_scale is not None else _pick("chunk_scale")

        required = {
            "normalization_rotation / norm_rot": norm_rot_val,
            "normalization_center / norm_center": norm_center_val,
            "normalization_scale / norm_scale": norm_scale_val,
            "chunk_rotation / chunk_rot": chunk_rot_val,
            "chunk_translation / chunk_trans": chunk_trans_val,
            "chunk_scale (or dataparser_scale=)": chunk_scale_val,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise KeyError(f"NormalizationChain.from_npz_metadata: missing required keys: {missing}")

        return cls(
            norm_scale=float(norm_scale_val),
            norm_center=np.asarray(norm_center_val, dtype=np.float32).reshape(3),
            norm_rot=np.asarray(norm_rot_val, dtype=np.float32).reshape(3, 3),
            chunk_scale=float(chunk_scale_val),
            chunk_rot=np.asarray(chunk_rot_val, dtype=np.float32).reshape(3, 3),
            chunk_trans=np.asarray(chunk_trans_val, dtype=np.float32).reshape(3),
        )


def transform_markers_to_chunk_local(
    markers_global: dict[str, np.ndarray],
    s_chunk: float,
    R_chunk: np.ndarray,
    T_chunk: np.ndarray,
) -> np.ndarray:
    """Transform global marker coordinates into the chunk-local coordinate system.

    Args:
        markers_global: Dict mapping marker label -> (3,) global coordinate array.
        s_chunk: Chunk scale (global → chunk-local).
        R_chunk: Chunk rotation (3x3).
        T_chunk: Chunk translation (3,).

    Returns:
        (N, 3) array of chunk-local marker coordinates.
    """
    R_chunk_inv = R_chunk.T
    markers_chunk_local = []
    for _label, p_global in markers_global.items():
        markers_chunk_local.append((1 / s_chunk) * R_chunk_inv @ (p_global - T_chunk))
    return np.array(markers_chunk_local)


def calculate_scene_box(
    camera_to_worlds: np.ndarray,
    marker_positions: np.ndarray | None = None,
    aabb_scale: float = 1.5,
    center_override: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate the scene bounding box from camera positions and optionally marker positions.

    Args:
        camera_to_worlds: (N, 4, 4) camera-to-world matrices.
        marker_positions: Optional (M, 3) marker positions to include in AABB computation.
        aabb_scale: Scale factor applied to the bounding-box size.
        center_override: If given, use this as the AABB center instead of auto-computing.

    Returns:
        (3, 2) array [[xmin, xmax], [ymin, ymax], [zmin, zmax]].

    Raises:
        ValueError: If no points are available.
    """
    all_points = camera_to_worlds[:, :3, 3]  # Camera origins

    if marker_positions is not None and len(marker_positions) > 0:
        all_points = np.vstack((all_points, marker_positions))

    if len(all_points) == 0:
        raise ValueError("No points available to calculate scene box.")

    min_bound = all_points.min(axis=0)
    max_bound = all_points.max(axis=0)

    if center_override is None:
        center = (min_bound + max_bound) / 2.0
    else:
        center = np.asarray(center_override, dtype=np.float32)

    size = (max_bound - min_bound) * aabb_scale

    min_bound = center - size / 2.0
    max_bound = center + size / 2.0

    return np.stack([min_bound, max_bound], axis=0).T


def _global_to_scene_points(
    points_global: np.ndarray,
    s_chunk: float,
    R_chunk: np.ndarray,
    T_chunk: np.ndarray,
    R_norm: np.ndarray,
    center: np.ndarray,
    norm_scale: float,
) -> np.ndarray:
    """Apply the full global → scene transform used by ``bathyfacto-build-dataset``."""
    chain = NormalizationChain(
        norm_scale=norm_scale,
        norm_center=center,
        norm_rot=R_norm,
        chunk_scale=s_chunk,
        chunk_rot=R_chunk,
        chunk_trans=T_chunk,
    )
    return chain.global_to_scene(points_global)


def calculate_scene_box_from_mesh(
    mesh_path: Path,
    water_plane_normal_scene: np.ndarray,
    water_plane_d_scene: float,
    s_chunk: float,
    R_chunk: np.ndarray,
    T_chunk: np.ndarray,
    R_norm: np.ndarray,
    center: np.ndarray,
    norm_scale: float,
    buffer_m_global: float = 0.5,
) -> np.ndarray:
    """Compute scene_box in scene frame from a GT mesh AABB unioned with sampled water surface.

    The buffer is specified in *global* metres and converted to scene units via the
    isotropic distance factor ``1 / (s_chunk * norm_scale)``. Mesh vertices are read in
    the global frame, transformed through the full scene-normalization chain, and unioned
    with four samples of the water plane evaluated at the mesh's XY extent (in scene
    coords). The resulting axis-aligned bounding box is symmetrically padded by the
    converted buffer.

    Args:
        mesh_path: Path to GT mesh (.ply) with vertices in the global frame.
        water_plane_normal_scene: (3,) water-plane normal in scene frame.
        water_plane_d_scene: Plane offset constant (n·p + d = 0) in scene frame.
        s_chunk: Chunk scale (global → chunk-local).
        R_chunk: (3, 3) chunk rotation.
        T_chunk: (3,) chunk translation.
        R_norm: (3, 3) normalization rotation (typically R_chunk for georeferenced data).
        center: (3,) scene-translation offset (camera centroid post-rotation).
        norm_scale: Final isotropic scene scale factor.
        buffer_m_global: Symmetric absolute buffer in global metres. Defaults to 0.5 m.

    Returns:
        (3, 2) scene_box ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` in scene frame.

    Raises:
        FileNotFoundError: If ``mesh_path`` does not exist.
        ValueError: If the mesh has no vertices or if the water-plane normal is degenerate.
    """
    import open3d as o3d

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts_global = np.asarray(mesh.vertices, dtype=np.float64)
    if verts_global.size == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    verts_scene = _global_to_scene_points(verts_global, s_chunk, R_chunk, T_chunk, R_norm, center, norm_scale)

    mesh_min = verts_scene.min(axis=0)
    mesh_max = verts_scene.max(axis=0)

    nx, ny, nz = (float(c) for c in water_plane_normal_scene)
    if abs(nz) < 1e-6:
        raise ValueError("Water-plane normal has near-zero Z component in scene frame, cannot sample plane corners.")
    water_corners = np.array(
        [
            [x, y, -(water_plane_d_scene + nx * x + ny * y) / nz]
            for x in (mesh_min[0], mesh_max[0])
            for y in (mesh_min[1], mesh_max[1])
        ],
        dtype=np.float64,
    )

    union_points = np.vstack([verts_scene, water_corners])
    aabb_min = union_points.min(axis=0)
    aabb_max = union_points.max(axis=0)

    buffer_scene = float(buffer_m_global) / (float(s_chunk) * float(norm_scale))
    aabb_min = aabb_min - buffer_scene
    aabb_max = aabb_max + buffer_scene

    CONSOLE.log(
        f"bathyfacto  dataset: scene box from mesh {mesh_path.name}, {len(verts_global)} vertices, "
        f"buffer {buffer_m_global} m global ({buffer_scene:.6f} scene units), "
        f"box min {aabb_min.tolist()} max {aabb_max.tolist()}"
    )

    return np.stack([aabb_min, aabb_max], axis=0).T.astype(np.float32)


def _svd_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through ``points``: unit normal (z >= 0) and d with n.x + d = 0."""
    centroid = np.mean(points, axis=0)
    _, _, V = np.linalg.svd(points - centroid)
    normal = V[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, -np.dot(normal, centroid)


def fit_plane_to_points(
    points: np.ndarray,
    filter_outliers: bool = False,
    outlier_std_threshold: float = 2.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane to a set of 3D points using SVD.

    Optionally filters outlier points based on distance from the plane.

    Args:
        points: (N, 3) array of 3D points.
        filter_outliers: If True, filters points with distance > outlier_std_threshold * std.
        outlier_std_threshold: Number of standard deviations for outlier threshold.

    Returns:
        Tuple of:
        - normal: Normalized plane normal vector (3,).
        - d: Plane-equation constant (n·x + d = 0).
        - filter_mask: Boolean array, True for inliers.

    Raises:
        ValueError: If fewer than 3 points are provided.
    """
    if len(points) < 3:
        raise ValueError("At least 3 points are required to fit a plane.")

    normal, d = _svd_plane(points)
    all_inliers = np.ones(len(points), dtype=bool)
    if not filter_outliers:
        return normal, d, all_inliers

    distances = np.abs(np.dot(points, normal) + d)
    inlier_mask = distances <= outlier_std_threshold * np.std(distances)
    if np.sum(inlier_mask) < 3:
        CONSOLE.log(
            "[yellow]bathyfacto  dataset: fewer than 3 points left after outlier filtering, "
            "using the unfiltered points[/yellow]"
        )
        return normal, d, all_inliers

    normal_filtered, d_filtered = _svd_plane(points[inlier_mask])
    return normal_filtered, d_filtered, inlier_mask


def align_water_to_horizontal(normal: np.ndarray) -> np.ndarray:
    """Compute rotation matrix to align water plane normal to [0, 0, 1].

    This rotation makes the water surface horizontal (z=const), which simplifies
    refraction calculations and improves numerical stability.

    Non-georeferenced fallback only. For datasets whose Metashape XML contains a
    ``<chunk><transform>`` block (i.e. ``R_chunk != I``), ``bathyfacto-build-dataset``
    uses ``R_chunk`` directly as the scene normalization rotation instead, so the scene
    frame equals the true world frame up to a pure translation and uniform scale.
    This function is called only when ``R_chunk == I`` (no chunk transform present).

    Args:
        normal: Water plane normal vector (3,)

    Returns:
        R: 3x3 rotation matrix such that R @ normal ≈ [0, 0, 1]
    """
    normal = normal / np.linalg.norm(normal)
    target = np.array([0.0, 0.0, 1.0])

    if np.allclose(normal, target, atol=1e-6):
        return np.eye(3)

    if np.allclose(normal, -target, atol=1e-6):
        return np.diag([1.0, -1.0, -1.0])  # 180° around X

    v = np.cross(normal, target)
    c = np.dot(normal, target)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return R


def select_normalization_rotation(R_chunk: np.ndarray, original_water_normal: np.ndarray) -> np.ndarray:
    """Select the scene normalization rotation based on whether chunk transform is present.

    For georeferenced datasets (R_chunk != I), use R_chunk directly so that the scene
    frame equals the true world frame up to pure translation and uniform scale.
    For non-georeferenced datasets (R_chunk == I), fall back to a Rodrigues rotation that
    aligns the water plane normal to [0, 0, 1].

    Args:
        R_chunk: 3x3 chunk rotation from Metashape XML. Identity when no chunk transform present.
        original_water_normal: Water plane normal in chunk-local coordinates (before any rotation).

    Returns:
        R_norm: 3x3 normalization rotation as float32 ndarray.
    """
    if not np.allclose(R_chunk, np.eye(3)):
        CONSOLE.print("bathyfacto  dataset: scene normalization rotation set to chunk_rotation (georeferenced)")
        return np.asarray(R_chunk, dtype=np.float32)
    CONSOLE.print(
        "bathyfacto  dataset: scene normalization rotation set to align_water_to_horizontal "
        "(non-georeferenced fallback)"
    )
    return align_water_to_horizontal(original_water_normal).astype(np.float32)


def stratified_train_val_split(
    camera_to_worlds: np.ndarray,
    val_fraction: float = 0.1,
    nadir_deg: float = 20.0,
    azimuth_bins: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a train/val split with view-coverage across nadir/oblique and azimuth bins.

    Args:
        camera_to_worlds: (N, 4, 4) camera-to-world matrices.
        val_fraction: Fraction of cameras to use for validation.
        nadir_deg: Elevation threshold (degrees) below which a camera is considered nadir.
        azimuth_bins: Number of azimuth bins for stratification.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices) as integer arrays.

    Raises:
        ValueError: If fewer than 2 cameras are provided.
    """
    num_cameras = len(camera_to_worlds)
    if num_cameras < 2:
        raise ValueError("At least two cameras are required to create a train/val split.")

    val_count = max(1, int(np.floor(num_cameras * val_fraction)))
    val_count = min(val_count, num_cameras - 1)

    positions = camera_to_worlds[:, :3, 3]
    view_dirs = -camera_to_worlds[:, :3, 2]
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1, keepdims=True)

    down = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cos_down = np.clip(view_dirs @ down, -1.0, 1.0)
    elev_deg = np.degrees(np.arccos(cos_down))
    elev_bin = np.where(elev_deg <= nadir_deg, 0, 1)  # 0=nadir, 1=oblique

    azim = np.degrees(np.arctan2(positions[:, 1], positions[:, 0]))
    azim = (azim + 360.0) % 360.0
    az_bin = np.floor(azim / (360.0 / azimuth_bins)).astype(int)
    az_bin = np.clip(az_bin, 0, azimuth_bins - 1)

    rng = np.random.default_rng(seed)
    bins: dict[tuple[int, int], list[int]] = {}
    for idx, (e_bin, a_bin) in enumerate(zip(elev_bin, az_bin)):
        bins.setdefault((int(e_bin), int(a_bin)), []).append(idx)
    for key in bins:
        rng.shuffle(bins[key])

    # First pass: ensure coverage across available bins.
    selected: list[int] = []
    for key in sorted(bins.keys()):
        if bins[key]:
            selected.append(bins[key].pop(0))

    if len(selected) > val_count:
        selected = list(rng.choice(selected, size=val_count, replace=False))

    # Fill remaining by round-robin across bins with leftovers.
    bin_keys = [key for key in sorted(bins.keys()) if bins[key]]
    cursor = 0
    while len(selected) < val_count and bin_keys:
        key = bin_keys[cursor % len(bin_keys)]
        if bins[key]:
            selected.append(bins[key].pop(0))
            cursor += 1
        else:
            bin_keys.pop(cursor % len(bin_keys))

    selected_indices = np.array(sorted(set(selected)), dtype=int)
    train_indices = np.setdiff1d(np.arange(num_cameras, dtype=int), selected_indices)
    return train_indices, selected_indices
