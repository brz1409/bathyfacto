"""Tests for BathyFactoModel — single-sampler two-media NeRF."""

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.two_media_geometry import make_kinked_density_fn
from nerfstudio.models.bathyfacto import BathyFactoModel, BathyFactoModelConfig


def _make_model(
    disable_refraction: bool = False,
    num_train_data: int = 4,
) -> BathyFactoModel:
    """Create BathyFactoModel with minimal config and water surface metadata."""
    config = BathyFactoModelConfig(
        disable_refraction=disable_refraction,
        implementation="torch",
        use_appearance_embedding=False,
        disable_scene_contraction=True,
        num_proposal_iterations=1,
        proposal_net_args_list=[
            {"hidden_dim": 8, "log2_hashmap_size": 12, "num_levels": 2, "max_res": 64, "use_linear": False},
        ],
        num_proposal_samples_per_ray=(16,),
        num_nerf_samples_per_ray=8,
        predict_normals=False,
    )
    scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    metadata = {
        "water_surface": {
            "plane_model": {
                "normal": [0.0, 0.0, 1.0],
                "d": 0.5,
            }
        }
    }
    model = BathyFactoModel(
        config=config,
        scene_box=scene_box,
        num_train_data=num_train_data,
        metadata=metadata,
    )
    return model


def test_model_instantiation():
    """Model has proposal_sampler, field, water_plane_normal; does NOT have dual samplers."""
    model = _make_model()

    # Single proposal sampler
    assert hasattr(model, "proposal_sampler"), "Should have proposal_sampler"
    assert not hasattr(model, "air_proposal_sampler"), "Should NOT have air_proposal_sampler"
    assert not hasattr(model, "water_proposal_sampler"), "Should NOT have water_proposal_sampler"

    # Field is BathyField
    from nerfstudio.fields.bathy_field import BathyField

    assert isinstance(model.field, BathyField), f"Expected BathyField, got {type(model.field)}"

    # Water plane buffers
    assert hasattr(model, "water_plane_normal"), "Should have water_plane_normal buffer"
    assert hasattr(model, "water_plane_d"), "Should have water_plane_d buffer"
    assert model.water_plane_normal.shape == (1, 3)
    assert torch.allclose(model.water_plane_normal, torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(model.water_plane_d, torch.tensor([0.5]))


def test_param_groups():
    """get_param_groups returns expected keys."""
    model = _make_model()
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
    interface_param = torch.tensor([0.5, 0.5])  # [N] — interface at t=0.5
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


def _make_ray_bundle(model, n_rays=16):
    """Create a RayBundle with rays pointing downward through the water surface."""
    device = next(model.parameters()).device
    origins = torch.zeros(n_rays, 3, device=device)
    origins[:, 2] = 0.0  # above water (water plane at z = -0.5)
    directions = torch.zeros(n_rays, 3, device=device)
    directions[:, 2] = -1.0  # straight down
    medium_mask = torch.ones(n_rays, 1, device=device)
    return RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=torch.ones(n_rays, 1, device=device) * 1e-5,
        camera_indices=torch.zeros(n_rays, 1, dtype=torch.int32, device=device),
        nears=torch.full((n_rays, 1), 0.05, device=device),
        fars=torch.full((n_rays, 1), 32.0, device=device),
        metadata={model.config.medium_mask_key: medium_mask},
    )


def test_get_outputs_forward_pass():
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model)
    outputs = model.get_outputs(rb)
    assert "rgb" in outputs
    assert "depth" in outputs
    assert "expected_depth" in outputs
    assert "accumulation" in outputs
    assert outputs["rgb"].shape[-1] == 3
    assert not torch.isnan(outputs["rgb"]).any()


def test_get_outputs_no_nan_in_depth():
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model)
    outputs = model.get_outputs(rb)
    assert not torch.isnan(outputs["depth"]).any()
    assert not torch.isnan(outputs["expected_depth"]).any()


def test_get_loss_dict():
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model)
    outputs = model.get_outputs(rb)
    batch = {"image": torch.rand(rb.origins.shape[0], 3)}
    loss_dict = model.get_loss_dict(outputs, batch)
    assert "rgb_loss" in loss_dict
    assert "interlevel_loss" in loss_dict
    assert not torch.isnan(loss_dict["rgb_loss"])


def test_get_outputs_snapshot():
    """Coarse numeric snapshot: shapes, no NaN/Inf, and recorded mean values.

    Snapshot recorded 2026-05-12 on current (pre-refactor) bathy-markus-temp HEAD.
    If an intentional behavior change updates these constants, add a note explaining why.
    """
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model, n_rays=32)
    outputs = model.get_outputs(rb)

    # --- Key presence ---
    for key in ("rgb", "depth", "expected_depth", "accumulation"):
        assert key in outputs, f"missing output key: {key!r}"

    # --- Shape assertions ---
    assert outputs["rgb"].shape == (32, 3), f"rgb shape: {outputs['rgb'].shape}"
    assert outputs["depth"].shape == (32, 1), f"depth shape: {outputs['depth'].shape}"
    assert outputs["expected_depth"].shape == (32, 1)
    assert outputs["accumulation"].shape == (32, 1)

    # --- No NaN / Inf ---
    for key in ("rgb", "depth", "expected_depth", "accumulation"):
        assert not torch.isnan(outputs[key]).any(), f"NaN in {key!r}"
        assert not torch.isinf(outputs[key]).any(), f"Inf in {key!r}"

    # --- Coarse numeric snapshot (atol=1e-3 to survive minor float precision drift) ---
    # snapshot recorded 2026-05-12 on current code; intentional changes update here with a note
    torch.testing.assert_close(
        outputs["depth"].mean(),
        torch.tensor(0.86228061),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        outputs["expected_depth"].mean(),
        torch.tensor(0.46407634),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        outputs["accumulation"].mean(),
        torch.tensor(0.52525228),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        outputs["rgb"].mean(),
        torch.tensor(0.48588273),
        rtol=1e-3,
        atol=1e-3,
    )


def test_get_outputs_snapshot_no_refraction():
    """Snapshot variant with refraction disabled.

    Note: If Plan 03 removes the disable_refraction flag as dead code, drop this
    variant in that commit with a note rather than here.
    Snapshot recorded 2026-05-12 on current (pre-refactor) code.
    """
    torch.manual_seed(0)
    model = _make_model(disable_refraction=True)
    model.train()
    rb = _make_ray_bundle(model, n_rays=32)
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
