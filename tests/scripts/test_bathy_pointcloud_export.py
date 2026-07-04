from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.exporter.bathy_pointcloud_utils import (
    get_bathy_export_dataset,
    reconstruct_bathy_facto_points_from_depth,
    select_bathy_export_depth,
    transform_points_to_bathy_global_frame,
    update_bathy_point_reservoir,
)
from nerfstudio.scripts.exporter import get_parser_fn, resolve_bathy_output_dir


def _make_test_ray_bundle() -> RayBundle:
    return RayBundle(
        origins=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        directions=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        pixel_area=torch.ones((1, 1), dtype=torch.float32),
        camera_indices=torch.zeros((1, 1), dtype=torch.int64),
        nears=torch.zeros((1, 1), dtype=torch.float32),
        fars=torch.full((1, 1), 10.0, dtype=torch.float32),
        metadata={"medium_mask": torch.ones((1, 1), dtype=torch.float32)},
    )


def test_reconstruct_bathy_facto_points_from_depth():
    points, valid_water_rays, interface_param = reconstruct_bathy_facto_points_from_depth(
        _make_test_ray_bundle(),
        torch.tensor([[2.0]], dtype=torch.float32),
        water_plane_normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        water_plane_d=torch.tensor([0.0], dtype=torch.float32),
        scene_box_aabb=torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]], dtype=torch.float32),
        air_refractive_index=1.0,
        water_refractive_index=1.333,
        water_entry_epsilon=1e-4,
        water_mask_threshold=0.5,
    )

    assert valid_water_rays.tolist() == [True]
    torch.testing.assert_close(interface_param, torch.tensor([1.0], dtype=torch.float32))
    torch.testing.assert_close(points, torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32))


def test_transform_points_to_bathy_global_frame():
    # NormalizationChain invariant: norm_rot == chunk_rot for georeferenced datasets
    # (see `select_normalization_rotation` in bathy_normalization_utils). The exporter
    # inverse runs `from_npz_metadata`, which enforces this; use a shared rotation to
    # honor the contract.
    shared_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dataparser_outputs = SimpleNamespace(
        metadata={
            "normalization_rotation": shared_rotation,
            "normalization_center": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "normalization_scale": 10.0,
            "chunk_rotation": shared_rotation,
            "chunk_translation": np.array([5.0, 6.0, 7.0], dtype=np.float32),
        },
        dataparser_scale=2.0,
    )
    normalized_points = torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float32)

    global_points = transform_points_to_bathy_global_frame(normalized_points, dataparser_outputs)

    chunk_local = (normalized_points.numpy() * 10.0 + np.array([1.0, 2.0, 3.0], dtype=np.float32)) @ shared_rotation
    expected = 2.0 * (chunk_local @ shared_rotation.T) + np.array([5.0, 6.0, 7.0], dtype=np.float32)
    np.testing.assert_allclose(global_points.numpy(), expected, atol=1e-6)


def test_bathy_pointcloud_subcommand_is_registered():
    parser = get_parser_fn()
    help_text = parser.format_help()
    assert "bathy-pointcloud" in help_text


def test_resolve_bathy_output_dir_defaults_to_run_export_dir():
    load_config = Path("/tmp/run/config.yml")

    output_dir = resolve_bathy_output_dir(load_config, None)

    assert output_dir == Path("/tmp/run/exports/bathy-pointcloud")


def test_resolve_bathy_output_dir_keeps_explicit_override():
    load_config = Path("/tmp/run/config.yml")
    explicit = Path("/tmp/custom/export")

    output_dir = resolve_bathy_output_dir(load_config, explicit)

    assert output_dir == explicit


def test_update_bathy_point_reservoir_limits_num_points_and_preserves_pairs():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    points_a = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    colors_a = points_a + 100.0
    points_b = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    colors_b = points_b + 100.0

    reservoir_points, reservoir_colors, reservoir_keys = update_bathy_point_reservoir(
        None,
        None,
        None,
        points_a,
        colors_a,
        num_points=3,
        generator=generator,
    )
    reservoir_points, reservoir_colors, _ = update_bathy_point_reservoir(
        reservoir_points,
        reservoir_colors,
        reservoir_keys,
        points_b,
        colors_b,
        num_points=3,
        generator=generator,
    )

    assert reservoir_points.shape == (3, 3)
    assert reservoir_colors.shape == (3, 3)
    torch.testing.assert_close(reservoir_colors, reservoir_points + 100.0)


def test_get_bathy_export_dataset_uses_full_data_split():
    called_splits = []

    class FakeDataparser:
        def get_dataparser_outputs(self, split: str = "train"):
            called_splits.append(split)
            return "data-outputs"

    class FakeDataset:
        def __init__(self, dataparser_outputs, scale_factor):
            self.dataparser_outputs = dataparser_outputs
            self.scale_factor = scale_factor

    pipeline = SimpleNamespace(
        datamanager=SimpleNamespace(
            dataparser=FakeDataparser(),
            train_dataset=FakeDataset("train-outputs", 1.0),
            config=SimpleNamespace(camera_res_scale_factor=0.5),
        )
    )

    dataset, dataparser_outputs = get_bathy_export_dataset(pipeline)

    assert called_splits == ["data"]
    assert dataparser_outputs == "data-outputs"
    assert isinstance(dataset, FakeDataset)
    assert dataset.dataparser_outputs == "data-outputs"
    assert dataset.scale_factor == 0.5


def test_select_bathy_export_depth_prefers_expected_depth_for_bathyfacto(monkeypatch):
    outputs = {
        "depth": torch.tensor([[10.0]], dtype=torch.float32),
        "expected_depth": torch.tensor([[2.0]], dtype=torch.float32),
    }

    fake_model_cls = type("FakeBathyFactoModel", (), {})
    monkeypatch.setattr(
        "nerfstudio.exporter.bathy_pointcloud_utils.BathyFactoModel",
        fake_model_cls,
    )
    model = fake_model_cls()

    depth = select_bathy_export_depth(model, outputs)

    torch.testing.assert_close(depth, outputs["expected_depth"])
