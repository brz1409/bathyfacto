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

"""The refraction ablation (``disable_refraction``) on rays that cross the water at 35 degrees.

With refraction on, every sample past the water plane of a water ray sits on the Snell-bent
ray. With ``disable_refraction=True`` it sits on the straight camera ray. Samples above the
water, and every sample of a ray the medium mask marks as land, sit on the straight camera
ray in both modes.

The positions are read where the field receives them. The proposal networks are replaced by a
constant density so that the sample t-values depend only on the ray extent and not on the
proposal network, which would otherwise differ between the two modes.
"""

import math
from typing import Dict

import torch
from nerfstudio.cameras.rays import RayBundle

from bathyfacto.two_media_geometry import snell_refract

INCIDENCE_DEG = 35.0
N_WATER = 1.333
N_PER_KIND = 4
POS_ATOL = 1e-5
"""Float32 noise on positions of order one."""
MIN_SEPARATION = 1e-2
"""Refracted and straight water samples must be at least this far apart (scene units)."""


def _oblique_bundle() -> RayBundle:
    """Rays from the origin at 35 degrees incidence, fanned out in azimuth.

    The first half is marked water by the medium mask, the second half land. Every ray crosses
    the tiny model's water plane at z=-0.5 geometrically.
    """
    n_rays = 2 * N_PER_KIND
    theta = math.radians(INCIDENCE_DEG)
    azimuths = torch.linspace(0.0, 2.0 * math.pi, n_rays + 1)[:-1]
    directions = torch.stack(
        [
            math.sin(theta) * torch.cos(azimuths),
            math.sin(theta) * torch.sin(azimuths),
            torch.full_like(azimuths, -math.cos(theta)),
        ],
        dim=-1,
    )
    medium_mask = torch.zeros(n_rays, 1)
    medium_mask[:N_PER_KIND] = 1.0
    return RayBundle(
        origins=torch.zeros(n_rays, 3),
        directions=directions,
        pixel_area=torch.ones(n_rays, 1) * 1e-5,
        camera_indices=torch.zeros(n_rays, 1, dtype=torch.int32),
        nears=torch.full((n_rays, 1), 0.05),
        fars=torch.full((n_rays, 1), 32.0),
        metadata={"medium_mask": medium_mask},
    )


def _render(model, disable_refraction: bool) -> Dict[str, torch.Tensor]:
    """Run one eval forward pass and return the sample geometry the field was queried with."""
    model.config.disable_refraction = disable_refraction
    model.proposal_density_fns = [lambda positions: torch.ones(positions.shape[:-1] + (1,))]

    seen: Dict[str, torch.Tensor] = {}
    sampler_forward = model.proposal_sampler.forward
    field_get_outputs = model.field.get_outputs

    def capture_samples(*args, **kwargs):
        ray_samples, weights_list, ray_samples_list = sampler_forward(*args, **kwargs)
        seen["t"] = ((ray_samples.frustums.starts + ray_samples.frustums.ends) / 2).squeeze(-1)
        return ray_samples, weights_list, ray_samples_list

    def capture_positions(ray_samples, density_embedding=None, medium_flag=None):
        assert medium_flag is not None
        seen["positions"] = ray_samples.frustums.get_positions().detach().clone()
        seen["is_water"] = medium_flag.squeeze(-1).bool()
        return field_get_outputs(ray_samples, density_embedding=density_embedding, medium_flag=medium_flag)

    model.proposal_sampler.forward = capture_samples
    model.field.get_outputs = capture_positions
    try:
        with torch.no_grad():
            outputs = model.get_outputs(_oblique_bundle())
    finally:
        del model.proposal_sampler.forward
        del model.field.get_outputs
    seen.update({key: outputs[key] for key in ("interface_param", "entry_points", "refracted_dirs")})
    return seen


def _straight(bundle: RayBundle, t: torch.Tensor) -> torch.Tensor:
    directions = torch.nn.functional.normalize(bundle.directions, dim=-1)
    return bundle.origins.unsqueeze(1) + directions.unsqueeze(1) * t.unsqueeze(-1)


def _model(make_tiny_model):
    torch.manual_seed(0)
    model = make_tiny_model()
    model.eval()
    return model


def test_refracted_water_samples_follow_snells_law(make_tiny_model):
    """With refraction on, samples past the plane of a water ray lie on the Snell-bent ray."""
    model = _model(make_tiny_model)
    bundle = _oblique_bundle()
    run = _render(model, disable_refraction=False)

    expected_dirs = snell_refract(bundle.directions, model.water_plane_normal, 1.0, N_WATER)
    torch.testing.assert_close(run["refracted_dirs"], expected_dirs)
    # Independent check of the bend: sin(theta_t) = sin(theta_i) / n_water.
    sin_t = torch.linalg.norm(run["refracted_dirs"][:, :2], dim=-1)
    expected_sin_t = torch.full_like(sin_t, math.sin(math.radians(INCIDENCE_DEG)) / N_WATER)
    torch.testing.assert_close(sin_t, expected_sin_t)

    ip = run["interface_param"]  # [N, 1]
    water = run["is_water"]
    assert water[:N_PER_KIND].any(dim=-1).all(), "every water ray must have samples past the plane"
    bent = run["entry_points"].unsqueeze(1) + run["refracted_dirs"].unsqueeze(1) * (run["t"] - ip).unsqueeze(-1)
    torch.testing.assert_close(run["positions"][water], bent[water], atol=POS_ATOL, rtol=0.0)


def test_disabled_refraction_keeps_water_samples_on_the_straight_ray(make_tiny_model):
    model = _model(make_tiny_model)
    bundle = _oblique_bundle()
    run = _render(model, disable_refraction=True)

    torch.testing.assert_close(run["refracted_dirs"], torch.nn.functional.normalize(bundle.directions, dim=-1))
    assert run["is_water"][:N_PER_KIND].any(dim=-1).all(), "samples past the plane still count as water"
    torch.testing.assert_close(run["positions"], _straight(bundle, run["t"]), atol=POS_ATOL, rtol=0.0)


def test_refracted_and_straight_water_samples_are_clearly_apart(make_tiny_model):
    """Every refracted water sample more than 0.05 below the entry point is off the straight ray."""
    model = _model(make_tiny_model)
    bundle = _oblique_bundle()
    run = _render(model, disable_refraction=False)

    deep = run["is_water"] & ((run["t"] - run["interface_param"]) > 0.05)
    assert deep[:N_PER_KIND].any(dim=-1).all()
    offset = torch.linalg.norm(run["positions"] - _straight(bundle, run["t"]), dim=-1)
    assert bool((offset[deep] > MIN_SEPARATION).all()), f"smallest offset {offset[deep].min().item():.2e}"


def test_air_samples_lie_on_the_camera_ray_in_both_modes(make_tiny_model):
    """Above the water the two modes place samples with the same straight-ray mapping.

    The t-values of water rays are not shared between the modes (the ray ends where its water
    segment leaves the scene box, and that depends on the water direction), so the check is
    per sample against the camera ray and not a comparison of the two tensors.
    """
    model = _model(make_tiny_model)
    bundle = _oblique_bundle()
    for disable_refraction in (False, True):
        run = _render(model, disable_refraction=disable_refraction)
        air = ~run["is_water"]
        # On water rays the air samples are exactly the ones above the plane.
        water_ray_air = air[:N_PER_KIND]
        above_plane = run["t"][:N_PER_KIND] <= run["interface_param"][:N_PER_KIND]
        assert water_ray_air.any(dim=-1).all() and torch.equal(water_ray_air, above_plane)
        torch.testing.assert_close(run["positions"][air], _straight(bundle, run["t"])[air], atol=POS_ATOL, rtol=0.0)


def test_land_rays_are_identical_in_both_modes(make_tiny_model):
    """A ray the medium mask marks as land is never refracted, whatever the ablation flag."""
    model = _model(make_tiny_model)
    refracted = _render(model, disable_refraction=False)
    straight = _render(model, disable_refraction=True)

    land = slice(N_PER_KIND, None)
    assert not refracted["is_water"][land].any() and not straight["is_water"][land].any()
    torch.testing.assert_close(refracted["t"][land], straight["t"][land], atol=0.0, rtol=0.0)
    torch.testing.assert_close(refracted["positions"][land], straight["positions"][land], atol=0.0, rtol=0.0)
