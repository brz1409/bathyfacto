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

"""Tests for Snell's law refraction and TIR handling in two_media_geometry.

Tests analytically known refraction cases, TIR clamping, coplanarity, and the
disable_refraction passthrough using ``snell_refract`` from
``nerfstudio.model_components.two_media_geometry``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nerfstudio.model_components.two_media_geometry import snell_refract

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _angle_from_normal(direction: torch.Tensor, normal: torch.Tensor) -> float:
    """Angle in degrees between *direction* and the anti-normal (angle from vertical)."""
    d = F.normalize(direction.squeeze(), dim=-1)
    n = F.normalize(normal.squeeze(), dim=-1)
    cos = torch.clamp(torch.abs((d * n).sum()), 0.0, 1.0)
    return math.degrees(torch.acos(cos).item())


def _snell_expected_deg(theta_i_deg: float, n1: float, n2: float) -> float:
    """Analytical Snell's law refracted angle in degrees; NaN for TIR."""
    sin_t = (n1 / n2) * math.sin(math.radians(theta_i_deg))
    if abs(sin_t) > 1.0:
        return float("nan")
    return math.degrees(math.asin(sin_t))


def _make_direction(theta_deg: float, phi_deg: float = 0.0) -> torch.Tensor:
    """Direction going downward at *theta_deg* from the -Z axis (XZ plane)."""
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = -math.cos(theta)
    return torch.tensor([[x, y, z]], dtype=torch.float32)


# Horizontal water surface; IOR from the paper (air→water).
_NORMAL = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
_N1 = 1.0
_N2 = 1.333


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta_i", [0.0, 30.0, 45.0, 60.0])
def test_snell_refract_angle_matches_analytical(theta_i: float) -> None:
    """Refracted angle matches Snell's law within 0.05 deg for non-TIR incidence."""
    incident = _make_direction(theta_i)
    refracted = snell_refract(incident, _NORMAL, _N1, _N2)
    theta_t_exp = _snell_expected_deg(theta_i, _N1, _N2)
    theta_t_got = _angle_from_normal(refracted, _NORMAL)
    err = abs(theta_t_got - theta_t_exp)
    assert err < 0.05, f"theta_i={theta_i}: expected {theta_t_exp:.4f} deg got {theta_t_got:.4f} deg"


def test_snell_refract_vertical_incidence_no_bending() -> None:
    """Vertical incidence (0 deg) gives straight-through refraction — no angle change."""
    incident = _make_direction(0.0)
    refracted = snell_refract(incident, _NORMAL, _N1, _N2)
    expected = torch.tensor([[0.0, 0.0, -1.0]])
    torch.testing.assert_close(refracted, expected, atol=1e-5, rtol=0)


def test_snell_refract_near_tir_no_nan() -> None:
    """Near-TIR incidence (85 deg) is clamped to a valid direction — no NaN or Inf."""
    incident = _make_direction(85.0)
    refracted = snell_refract(incident, _NORMAL, _N1, _N2)
    assert not torch.isnan(refracted).any(), "NaN in refracted direction near TIR"
    assert not torch.isinf(refracted).any(), "Inf in refracted direction near TIR"
    assert refracted[0, 2].item() < 0, "Refracted direction must point downward"


@pytest.mark.parametrize("theta_i", [0.0, 30.0, 45.0])
def test_snell_refract_direction_points_downward(theta_i: float) -> None:
    """Refracted ray has a negative z-component (entering the water medium)."""
    incident = _make_direction(theta_i)
    refracted = snell_refract(incident, _NORMAL, _N1, _N2)
    assert refracted[0, 2].item() < 0, f"theta_i={theta_i}: refracted z must be negative"


def test_snell_refract_disable_refraction_passthrough() -> None:
    """With disable_refraction=True the function returns normalised incident directions."""
    incident = _make_direction(30.0)
    straight = snell_refract(incident, _NORMAL, _N1, _N2, disable_refraction=True)
    expected = F.normalize(incident, dim=-1)
    torch.testing.assert_close(straight, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("theta_i", [15.0, 30.0, 45.0, 60.0])
def test_snell_refract_coplanarity(theta_i: float) -> None:
    """Incident, refracted, and normal lie in the same plane (Snell coplanarity constraint)."""
    incident = _make_direction(theta_i)
    refracted = snell_refract(incident, _NORMAL, _N1, _N2)
    plane_normal = torch.cross(incident.squeeze(), _NORMAL.squeeze())
    plane_normal = F.normalize(plane_normal, dim=0)
    deviation = abs(torch.dot(refracted.squeeze(), plane_normal).item())
    assert deviation < 1e-5, f"theta_i={theta_i}: out-of-plane deviation={deviation:.2e}"
