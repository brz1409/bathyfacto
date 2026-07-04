"""Tests for BathyField — NerfactoField with 1-bit medium conditioning."""

import torch

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.bathy_field import BathyField


def _make_ray_samples(n_rays: int = 4, n_samples: int = 8, device: str = "cpu") -> RaySamples:
    """Create minimal RaySamples with Frustums and camera_indices."""
    origins = torch.rand(n_rays, n_samples, 3, device=device)
    directions = torch.randn(n_rays, n_samples, 3, device=device)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    starts = torch.rand(n_rays, n_samples, 1, device=device)
    ends = starts + 0.1
    pixel_area = torch.ones(n_rays, n_samples, 1, device=device)
    frustums = Frustums(
        origins=origins,
        directions=directions,
        starts=starts,
        ends=ends,
        pixel_area=pixel_area,
    )
    camera_indices = torch.zeros(n_rays, n_samples, 1, dtype=torch.long, device=device)
    deltas = ends - starts
    return RaySamples(frustums=frustums, camera_indices=camera_indices, deltas=deltas)


def _make_field(appearance_embedding_dim: int = 32, num_images: int = 4) -> BathyField:
    """Create BathyField with default test parameters."""
    aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    return BathyField(
        aabb=aabb,
        num_images=num_images,
        geo_feat_dim=15,
        appearance_embedding_dim=appearance_embedding_dim,
        implementation="torch",
    )


def test_bathy_field_instantiation():
    """mlp_head.in_dim == 15 (geo_feat) + 16 (SH levels=4) + 32 (appearance) + 1 (medium) = 64."""
    field = _make_field(appearance_embedding_dim=32)
    expected_in_dim = 15 + 16 + 32 + 1  # geo_feat + SH + appearance + medium
    assert field.mlp_head.in_dim == expected_in_dim, (
        f"Expected mlp_head.in_dim={expected_in_dim}, got {field.mlp_head.in_dim}"
    )


def test_bathy_field_density_unchanged():
    """Density output shape is (n_rays, n_samples, 1) — unchanged from NerfactoField."""
    field = _make_field()
    field.eval()
    ray_samples = _make_ray_samples(n_rays=4, n_samples=8)
    density, geo_feat = field.get_density(ray_samples)
    assert density.shape == (4, 8, 1), f"Expected density shape (4, 8, 1), got {density.shape}"
    assert geo_feat.shape == (4, 8, 15), f"Expected geo_feat shape (4, 8, 15), got {geo_feat.shape}"


def test_bathy_field_medium_flag_changes_rgb():
    """Air flag (0) vs water flag (1) must produce different RGB outputs."""
    field = _make_field()
    field.eval()
    ray_samples = _make_ray_samples(n_rays=4, n_samples=8)

    density, geo_feat = field.get_density(ray_samples)

    air_flag = torch.zeros(4, 8, 1)
    water_flag = torch.ones(4, 8, 1)

    out_air = field.get_outputs(ray_samples, density_embedding=geo_feat, medium_flag=air_flag)
    out_water = field.get_outputs(ray_samples, density_embedding=geo_feat, medium_flag=water_flag)

    rgb_air = out_air[FieldHeadNames.RGB]
    rgb_water = out_water[FieldHeadNames.RGB]

    assert rgb_air.shape == (4, 8, 3)
    assert rgb_water.shape == (4, 8, 3)
    # With randomly initialized weights, different medium flags should produce different outputs
    assert not torch.allclose(rgb_air, rgb_water, atol=1e-6), "Air and water RGB should differ"


def test_bathy_field_no_appearance_embedding():
    """Field works correctly with appearance_embedding_dim=0."""
    field = _make_field(appearance_embedding_dim=0)
    field.eval()
    expected_in_dim = 15 + 16 + 0 + 1  # geo_feat + SH + no appearance + medium
    assert field.mlp_head.in_dim == expected_in_dim

    ray_samples = _make_ray_samples(n_rays=4, n_samples=8)
    density, geo_feat = field.get_density(ray_samples)
    out = field.get_outputs(ray_samples, density_embedding=geo_feat)
    assert out[FieldHeadNames.RGB].shape == (4, 8, 3)


def test_bathy_field_no_medium_flag():
    """BathyField with use_medium_flag=False has same MLP input dim as NerfactoField."""
    aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    field = BathyField(aabb, num_images=1, use_medium_flag=False, implementation="torch")
    # MLP input should NOT include the +1 medium flag dimension
    base_in_dim = field.direction_encoding.get_out_dim() + field.geo_feat_dim + field.appearance_embedding_dim
    assert field.mlp_head.in_dim == base_in_dim
