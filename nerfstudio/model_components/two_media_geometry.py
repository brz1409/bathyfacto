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

"""Two-media (air/water) ray geometry: water-plane intersection, Snell refraction, kinked-ray correction."""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

INTERFACE_PARALLEL_EPS: float = 1e-6
"""Threshold below which a ray's direction is considered parallel to the water plane."""


def signed_distance_to_water(positions: Tensor, plane_normal: Tensor, plane_d: Tensor) -> Tensor:
    """Signed distance from positions to water plane.

    Positive values are above water (air side).

    Args:
        positions: (N, 3) world positions.
        plane_normal: (1, 3) unit normal pointing into air.
        plane_d: (1,) plane offset so that normal · x + d = 0 at the surface.

    Returns:
        (N,) signed distances.
    """
    return (positions @ plane_normal.T).squeeze(-1) + plane_d


def snell_refract(
    incident_dirs: Tensor,
    plane_normal: Tensor,
    n1: float,
    n2: float,
    eps: float = 1e-7,
    disable_refraction: bool = False,
) -> Tensor:
    """Snell's law refraction with TIR handling.

    Args:
        incident_dirs: (N, 3) incoming ray directions (need not be normalised).
        plane_normal: (1, 3) water-surface normal (unit, pointing into air).
        n1: refractive index of the incident medium (air).
        n2: refractive index of the transmitted medium (water).
        eps: numerical epsilon for TIR clamping.
        disable_refraction: if True, return normalised incident directions unchanged.

    Returns:
        (N, 3) normalised refracted (or straight-through) directions.
    """
    if disable_refraction:
        return F.normalize(incident_dirs, dim=-1)

    normal = F.normalize(plane_normal, dim=-1)
    incident = F.normalize(incident_dirs, dim=-1)
    cos_i = -(incident * normal).sum(dim=-1, keepdim=True)
    normal_facing = torch.where(cos_i < 0, -normal, normal)
    cos_i = torch.abs(cos_i)

    eta = float(n1) / float(n2)
    sin2_t = (eta**2) * (1.0 - cos_i**2)
    is_tir = sin2_t > (1.0 - eps)
    sin2_t = sin2_t.clamp(max=1.0 - eps)

    cos_t = torch.sqrt(1.0 - sin2_t)
    refracted = eta * incident + (eta * cos_i - cos_t) * normal_facing
    reflected = incident - 2.0 * (incident * normal_facing).sum(dim=-1, keepdim=True) * normal_facing
    out = torch.where(is_tir, reflected, refracted)
    return F.normalize(out, dim=-1)


def water_plane_intersection(
    origins: Tensor,
    directions: Tensor,
    plane_normal: Tensor,
    plane_d: Tensor,
    parallel_eps: float = INTERFACE_PARALLEL_EPS,
) -> Tuple[Tensor, Tensor]:
    """Intersect rays with the water plane.

    Args:
        origins: (N, 3) ray origins.
        directions: (N, 3) normalised ray directions.
        plane_normal: (1, 3) unit water-surface normal.
        plane_d: (1,) plane offset (normal · x + d = 0 at surface).
        parallel_eps: threshold below which the ray is considered parallel to the plane.

    Returns:
        interface_param: (N,) t-values of intersection (set to +inf for parallel rays; callers
            typically clamp this to a far-plane sentinel).
        dir_dot_n: (N,) dot product of direction and normal (for caller reuse).
    """
    signed_dist = signed_distance_to_water(origins, plane_normal, plane_d)
    dir_dot_n = (directions @ plane_normal.T).squeeze(-1)
    # Rays nearly parallel to the plane never hit it; push param beyond scene
    interface_param = torch.where(
        dir_dot_n.abs() > parallel_eps,
        -signed_dist / dir_dot_n,
        torch.full_like(signed_dist, float("inf")),
    )
    return interface_param, dir_dot_n


def make_kinked_density_fn(
    base_density_fn: Callable[[Tensor], Tensor],
    origins: Tensor,
    air_dirs: Tensor,
    interface_param: Tensor,
    interface_pts: Tensor,
    water_dirs: Tensor,
) -> Callable[[Tensor], Tensor]:
    """Wrap a density function to correct water-segment positions via Snell refraction.

    The proposal sampler evaluates density along the straight air ray.  For samples
    beyond the water interface (t > interface_param), this wrapper remaps their
    positions onto the refracted (kinked) ray before querying the density field.

    Args:
        base_density_fn: callable accepting (N, S, 3) positions, returning density.
        origins: (N, 3) ray origins.
        air_dirs: (N, 3) normalised air-segment directions.
        interface_param: (N,) t-value of each ray's water-plane intersection.
        interface_pts: (N, 3) entry points on the water surface.
        water_dirs: (N, 3) normalised refracted directions.

    Returns:
        Wrapped density function with identical call signature.
    """

    def kinked_fn(positions: Tensor) -> Tensor:
        # Recover t: project (pos - origin) onto air_dir
        delta = positions - origins.unsqueeze(-2)
        t = (delta * air_dirs.unsqueeze(-2)).sum(-1, keepdim=True)

        # Which samples are in water?
        ip = interface_param.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
        is_water = t > ip

        # Correct water positions: interface_pts + water_dirs * (t - ip)
        water_t = t - ip
        water_pos = interface_pts.unsqueeze(-2) + water_dirs.unsqueeze(-2) * water_t
        corrected = torch.where(is_water, water_pos, positions)

        return base_density_fn(corrected)

    return kinked_fn


def gt_mesh_raycast(
    origins: Tensor,
    directions: Tensor,
    scene: "o3d.t.geometry.RaycastingScene",
    mesh_t: "o3d.t.geometry.TriangleMesh",
    fallback_normal: Tensor,
    fallback_d: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Intersect rays against a GT mesh; fall back to flat-plane intersection on misses.

    Performs CPU-side Open3D raycasting and returns per-ray water-surface interface
    parameters, entry points, and surface normals.  All returned tensors are on the
    same device as ``origins``.

    Args:
        origins: (N, 3) ray origins in scene frame (any device).
        directions: (N, 3) ray directions in scene frame (need not be unit length).
        scene: Pre-built ``o3d.t.geometry.RaycastingScene`` in scene frame.
        mesh_t: ``o3d.t.geometry.TriangleMesh`` used for per-vertex normal lookup.
        fallback_normal: (1, 3) unit water-plane normal for miss rays.
        fallback_d: (1,) water-plane offset for miss rays (normal · x + d = 0).

    Returns:
        interface_param: (N,) t-values of surface intersection (scene frame).
        entry_points: (N, 3) water-surface entry points (scene frame).
        normals: (N, 3) per-ray surface normals — barycentrically-interpolated
            per-vertex normals for GT hits, fallback_normal broadcast for misses.
        hit_mask: (N,) bool — True where GT mesh was hit; False where fallback used.
    """
    import numpy as np
    import open3d as o3d  # noqa: F811 — runtime import; TYPE_CHECKING guard above is for type checkers

    dev = origins.device

    # --- CPU/numpy bridge (pitfall 2: GPU tensors cannot call .numpy()) ---
    o_np = origins.detach().cpu().numpy().astype(np.float32)
    d_np = directions.detach().cpu().numpy().astype(np.float32)

    # Normalise directions before packing rays (pitfall 3: t_hit measured along provided direction)
    d_norms = np.linalg.norm(d_np, axis=1, keepdims=True).clip(1e-8)
    d_np = d_np / d_norms

    rays_np = np.concatenate([o_np, d_np], axis=1)  # (N, 6)
    rays_o3d = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)  # pitfall 5: must be Float32
    res = scene.cast_rays(rays_o3d)

    t_np = res["t_hit"].numpy()  # (N,) float32; +inf = miss
    hit_np = np.isfinite(t_np)  # (N,) bool
    pid_np = res["primitive_ids"].numpy()  # (N,) uint32
    uvs_np = res["primitive_uvs"].numpy()  # (N, 2) float32 — (u, v)

    # --- Barycentrically-interpolated per-vertex normals for hit rays ---
    has_vnormals = "normals" in mesh_t.vertex
    N = origins.shape[0]
    normals_np = np.zeros((N, 3), dtype=np.float32)

    if hit_np.any():
        tri_indices = mesh_t.triangle.indices.numpy()  # (F, 3) int32/uint32
        if has_vnormals:
            vn = mesh_t.vertex.normals.numpy()  # (V, 3) float32
            fv = tri_indices[pid_np[hit_np]]  # (K, 3) vertex indices for hit rays
            u = uvs_np[hit_np, 0:1]  # (K, 1)
            v = uvs_np[hit_np, 1:2]  # (K, 1)
            w = 1.0 - u - v
            interp = w * vn[fv[:, 0]] + u * vn[fv[:, 1]] + v * vn[fv[:, 2]]  # (K, 3)
            norms = np.linalg.norm(interp, axis=1, keepdims=True).clip(1e-8)
            normals_np[hit_np] = interp / norms
        else:
            # No vertex normals — use per-face geometric normals as fallback
            normals_np[hit_np] = res["primitive_normals"].numpy()[hit_np]

    # --- Torch tensors on original device ---
    t_torch = torch.from_numpy(t_np).to(device=dev)
    hit_mask = torch.from_numpy(hit_np).to(device=dev)
    normals = torch.from_numpy(normals_np).to(device=dev)

    # --- Miss fallback: water_plane_intersection for rays that missed the mesh ---
    miss_mask = ~hit_mask
    interface_param = t_torch.clone()
    if miss_mask.any():
        ip_fallback, _ = water_plane_intersection(
            origins[miss_mask], directions[miss_mask], fallback_normal, fallback_d
        )
        interface_param[miss_mask] = ip_fallback
        # Broadcast fallback normal to all miss rays
        n_misses = int(miss_mask.sum().item())
        normals[miss_mask] = fallback_normal.expand(n_misses, -1)

    # --- Entry points from final interface_param ---
    # Use normalised directions (consistent with t_hit measurement)
    d_norm_torch = torch.from_numpy(d_np).to(device=dev)
    entry_points = origins + d_norm_torch * interface_param.unsqueeze(-1)

    return interface_param, entry_points, normals, hit_mask
