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

"""Tests for BathyFactoModel, the single-sampler two-media NeRF."""

import torch
from conftest import down_ray_bundle
from nerfstudio.cameras.rays import RayBundle

from bathyfacto.bathyfacto_model import BathyFactoModelConfig
from bathyfacto.two_media_geometry import make_kinked_density_fn


def test_model_instantiation(make_tiny_model):
    """Model has a single proposal_sampler, field, and water_plane_normal."""
    model = make_tiny_model()

    assert hasattr(model, "proposal_sampler"), "Should have proposal_sampler"

    # Field is BathyField
    from bathyfacto.bathyfacto_field import BathyField

    assert isinstance(model.field, BathyField), f"Expected BathyField, got {type(model.field)}"

    # Water plane buffers
    assert hasattr(model, "water_plane_normal"), "Should have water_plane_normal buffer"
    assert hasattr(model, "water_plane_d"), "Should have water_plane_d buffer"
    assert model.water_plane_normal.shape == (1, 3)
    assert torch.allclose(model.water_plane_normal, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(model.water_plane_d, torch.tensor([0.5]))


def test_param_groups(make_tiny_model):
    """get_param_groups returns expected keys."""
    model = make_tiny_model()
    param_groups = model.get_param_groups()
    assert "proposal_networks" in param_groups, "Missing 'proposal_networks' key"
    assert "fields" in param_groups, "Missing 'fields' key"
    assert len(param_groups["proposal_networks"]) > 0, "proposal_networks should have parameters"
    assert len(param_groups["fields"]) > 0, "fields should have parameters"


def test_kinked_density_fn_air_positions_unchanged():
    """Positions with t < interface_param are passed through unmodified to base_fn."""
    n_rays = 2
    origins = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # [N, 3]
    air_dirs = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])  # [N, 3]
    interface_param = torch.tensor([0.5, 0.5])  # [N], interface at t=0.5
    interface_pts = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])  # [N, 3]
    water_dirs = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])  # [N, 3]

    # Create positions in air (t < 0.5): origin + dir * t, with t = 0.1, 0.2, 0.3
    t_vals = torch.tensor([0.1, 0.2, 0.3]).unsqueeze(0).expand(n_rays, -1)  # [N, S]
    positions = origins.unsqueeze(1) + air_dirs.unsqueeze(1) * t_vals.unsqueeze(-1)  # [N, S, 3]

    captured = []

    def recording_fn(pos):
        captured.append(pos.clone())
        return torch.ones(*pos.shape[:-1], 1)

    kinked_fn = make_kinked_density_fn(recording_fn, origins, air_dirs, interface_param, interface_pts, water_dirs)
    kinked_fn(positions)

    assert len(captured) == 1
    # Air positions should be unchanged
    assert torch.allclose(captured[0], positions, atol=1e-6), "Air positions should pass through unchanged"


def test_kinked_density_fn_water_positions_corrected():
    """Positions with t > interface_param are remapped to the refracted ray."""
    n_rays = 2
    origins = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    air_dirs = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    interface_param = torch.tensor([0.5, 0.5])
    interface_pts = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    # Use a different water direction to verify remapping
    water_dirs = torch.nn.functional.normalize(torch.tensor([[0.1, 0.0, -1.0], [0.1, 0.0, -1.0]]), dim=-1)

    # Create positions in water (t > 0.5): t = 0.7, 0.9
    t_vals = torch.tensor([0.7, 0.9]).unsqueeze(0).expand(n_rays, -1)
    positions = origins.unsqueeze(1) + air_dirs.unsqueeze(1) * t_vals.unsqueeze(-1)

    captured = []

    def recording_fn(pos):
        captured.append(pos.clone())
        return torch.ones(*pos.shape[:-1], 1)

    kinked_fn = make_kinked_density_fn(recording_fn, origins, air_dirs, interface_param, interface_pts, water_dirs)
    kinked_fn(positions)

    assert len(captured) == 1
    # Water positions should be corrected: interface_pts + water_dirs * (t - ip)
    water_t = t_vals.unsqueeze(-1) - 0.5  # [N, S, 1]
    expected = interface_pts.unsqueeze(1) + water_dirs.unsqueeze(1) * water_t
    assert torch.allclose(captured[0], expected, atol=1e-5), (
        f"Water positions should be remapped.\nGot: {captured[0]}\nExpected: {expected}"
    )


def test_get_loss_dict(make_tiny_model):
    model = make_tiny_model()
    model.train()
    rb = down_ray_bundle(model, 16, all_water=True)
    outputs = model.get_outputs(rb)
    batch = {"image": torch.rand(rb.origins.shape[0], 3)}
    loss_dict = model.get_loss_dict(outputs, batch)
    assert "rgb_loss" in loss_dict
    assert "interlevel_loss" in loss_dict
    assert not torch.isnan(loss_dict["rgb_loss"])


def test_get_outputs_snapshot_no_refraction(make_tiny_model):
    """Snapshot variant with refraction disabled.

    Snapshot recorded 2026-05-12.
    """
    torch.manual_seed(0)
    model = make_tiny_model(disable_refraction=True)
    model.train()
    rb = down_ray_bundle(model, 32, all_water=True)
    outputs = model.get_outputs(rb)

    for key in ("rgb", "depth", "expected_depth", "accumulation"):
        assert key in outputs
        assert not torch.isnan(outputs[key]).any(), f"NaN in {key!r}"
        assert not torch.isinf(outputs[key]).any(), f"Inf in {key!r}"

    assert outputs["rgb"].shape == (32, 3)
    assert outputs["depth"].shape == (32, 1)

    # snapshot recorded 2026-05-12; update with a note if behavior intentionally changes
    torch.testing.assert_close(
        outputs["depth"].mean(),
        torch.tensor(0.86228061),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        outputs["accumulation"].mean(),
        torch.tensor(0.52525228),
        rtol=1e-3,
        atol=1e-3,
    )


def test_config_only_adds_two_media_physics_fields():
    """The released model config carries only two-media physics and export routing on top of
    Nerfacto, none of the training-time diagnostics callbacks."""
    from nerfstudio.models.nerfacto import NerfactoModelConfig

    added = set(BathyFactoModelConfig.__dataclass_fields__) - set(NerfactoModelConfig.__dataclass_fields__)
    assert added == {
        "air_refractive_index",
        "water_refractive_index",
        "water_entry_epsilon",
        "water_mask_threshold",
        "valid_mask_key",
        "disable_refraction",
        "kinked_density_respects_interface_hits",
    }


def test_medium_mask_key_is_the_datamanager_constant_not_a_config_field(make_tiny_model):
    """The model reads the medium mask under the one key the datamanager publishes.

    A configurable key could name a key nothing writes: training and eval would then silently
    fall back to the geometric water plane while the exporter still used the mask.
    """
    import pytest

    from bathyfacto.bathyfacto_datamanager import attach_medium_mask

    with pytest.raises(TypeError):
        BathyFactoModelConfig(medium_mask_key="water_mask")  # type: ignore[call-arg]

    model = make_tiny_model()
    model.eval()
    n_rays = 4
    land = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    bundle = RayBundle(
        origins=torch.zeros(n_rays, 3),
        directions=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n_rays, 1),
        pixel_area=torch.ones(n_rays, 1) * 1e-5,
        camera_indices=torch.zeros(n_rays, 1, dtype=torch.int32),
        nears=torch.full((n_rays, 1), 0.05),
        fars=torch.full((n_rays, 1), 32.0),
    )
    bundle = attach_medium_mask(bundle, land.bool())
    with torch.no_grad():
        outputs = model.get_outputs(bundle)
    assert outputs["interface_hit_mask"].squeeze(-1).tolist() == [True, False, True, False]
