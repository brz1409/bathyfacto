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

"""Two-media (air/water) ray geometry: water-plane intersection, Snell refraction, kinked-ray correction."""

from __future__ import annotations

from typing import Callable, Optional

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
) -> Tensor:
    """Intersect rays with the water plane.

    Args:
        origins: (N, 3) ray origins.
        directions: (N, 3) normalised ray directions.
        plane_normal: (1, 3) unit water-surface normal.
        plane_d: (1,) plane offset (normal · x + d = 0 at surface).
        parallel_eps: threshold below which the ray is considered parallel to the plane.

    Returns:
        (N,) t-values of intersection (set to +inf for parallel rays; callers typically clamp
        this to a far-plane sentinel).
    """
    signed_dist = signed_distance_to_water(origins, plane_normal, plane_d)
    dir_dot_n = (directions @ plane_normal.T).squeeze(-1)
    # Rays nearly parallel to the plane never hit it; push param beyond scene
    return torch.where(
        dir_dot_n.abs() > parallel_eps,
        -signed_dist / dir_dot_n,
        torch.full_like(signed_dist, float("inf")),
    )


def make_kinked_density_fn(
    base_density_fn: Callable[[Tensor], Tensor],
    origins: Tensor,
    air_dirs: Tensor,
    interface_param: Tensor,
    interface_pts: Tensor,
    water_dirs: Tensor,
    interface_hits: Optional[Tensor] = None,
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
        interface_hits: optional (N,) bool mask of rays that actually enter the water.
            When given, only those rays get their samples refracted, matching how the final
            sample positions are corrected in ``BathyFactoModel.get_outputs``. When ``None``,
            refraction is applied to every ray whose geometric t exceeds ``interface_param``,
            including land rays excluded by the medium mask. The proposal network then sees a
            different geometry than the field it proposes samples for.

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
        if interface_hits is not None:
            is_water = is_water & interface_hits.unsqueeze(-1).unsqueeze(-1)

        # Correct water positions: interface_pts + water_dirs * (t - ip)
        water_t = t - ip
        water_pos = interface_pts.unsqueeze(-2) + water_dirs.unsqueeze(-2) * water_t
        corrected = torch.where(is_water, water_pos, positions)

        return base_density_fn(corrected)

    return kinked_fn
