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

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tyro
from nerfstudio.cameras.rays import RayBundle

import bathyfacto.export as export_module
from bathyfacto.export import (
    ExportBathyPointCloud,
    build_export_to_train_index_map,
    get_bathy_export_dataset,
    reconstruct_bathy_facto_points_from_depth,
    reconstruct_bathy_points,
    resolve_bathy_output_dir,
    select_bathy_export_depth,
    transform_points_to_bathy_global_frame,
    update_bathy_point_reservoir,
)


@pytest.fixture
def recorder(record_console):
    return record_console(export_module)


def test_reconstruct_bathy_facto_points_from_depth():
    """A nadir ray entering flat water at t=1 with depth 2 lands 1 below the entry point."""
    points, valid_water_rays = reconstruct_bathy_facto_points_from_depth(
        torch.tensor([[2.0]], dtype=torch.float32),
        interface_param=torch.tensor([1.0], dtype=torch.float32),
        entry_points=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        refracted_dirs=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        interface_hits=torch.tensor([True]),
        scene_box_aabb=torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]], dtype=torch.float32),
        water_entry_epsilon=1e-4,
    )

    assert valid_water_rays.tolist() == [True]
    torch.testing.assert_close(points, torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32))


def test_reconstruct_bathy_points_uses_the_configured_water_entry_epsilon(make_tiny_model):
    """``config.water_entry_epsilon`` is the epsilon the exporter applies.

    Oblique rays cross the water plane at z=-0.5 at t=0.52, render a depth of 0.8, and have
    about 0.5 of refracted path left inside the scene box. With the default epsilon the point
    is moved onto the refracted ray. An epsilon longer than that refracted path rejects it,
    so the point stays on the straight camera ray.
    """
    num_rays = 4
    directions = torch.nn.functional.normalize(torch.tensor([[0.3, 0.0, -1.0]]).repeat(num_rays, 1), dim=-1)
    depth = 0.8
    straight_points = directions * depth

    def export(water_entry_epsilon: float) -> torch.Tensor:
        ray_bundle = RayBundle(
            origins=torch.zeros(num_rays, 3),
            directions=directions.clone(),
            pixel_area=torch.ones(num_rays, 1),
            camera_indices=torch.zeros(num_rays, 1, dtype=torch.int64),
        )
        model = make_tiny_model(water_entry_epsilon=water_entry_epsilon)
        model.eval()
        with torch.no_grad():
            outputs = model(ray_bundle)
        outputs = {
            **outputs,
            "expected_depth": torch.full((num_rays, 1), depth),
            "accumulation": torch.ones(num_rays, 1),
        }
        points, _ = reconstruct_bathy_points(model, ray_bundle, outputs)
        return points

    refracted = export(1e-4)
    assert refracted.shape == (num_rays, 3)
    assert not torch.allclose(refracted, straight_points, atol=1e-3), "default epsilon must refract the point"
    torch.testing.assert_close(export(10.0), straight_points)


def test_reconstruct_bathy_points_refracts_water_rays_and_keeps_land_rays_straight(make_tiny_model):
    """End-to-end on a tiny model: the model renders, the exporter places each point.

    Row 0 is water, row 1 land. The plane sits at z=-0.5. A water ray whose depth lies past the
    plane is placed on the refracted ray, every other ray on its straight camera ray. The expected
    points are computed here with vector Snell, independent of the package geometry helpers.
    """
    torch.manual_seed(0)
    model = make_tiny_model()
    model.eval()
    height, width = 2, 3
    tilts = torch.tensor([[-0.2, 0.0, 0.25], [0.1, -0.15, 0.3]])
    directions = torch.nn.functional.normalize(
        torch.stack([tilts, 0.5 * tilts, -torch.ones(height, width)], dim=-1), dim=-1
    )
    medium_mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]).unsqueeze(-1)
    ray_bundle = RayBundle(
        origins=torch.zeros(height, width, 3),
        directions=directions.clone(),
        pixel_area=torch.ones(height, width, 1),
        camera_indices=torch.zeros(height, width, 1, dtype=torch.int64),
        metadata={"medium_mask": medium_mask},
    )
    with torch.no_grad():
        outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle)
    depth = torch.tensor([[0.3, 0.8, 0.9], [0.8, 0.6, 0.3]]).unsqueeze(-1)
    outputs = {**outputs, "expected_depth": depth, "accumulation": torch.ones(height, width, 1)}

    points, colors = reconstruct_bathy_points(model, ray_bundle, outputs)

    d = directions.reshape(-1, 3).double()
    depth_flat = depth.reshape(-1).double()
    expected = d * depth_flat.unsqueeze(-1)
    eta = 1.0 / 1.333
    for i in range(3):  # water row
        t_plane = 0.5 / -d[i, 2]
        if depth_flat[i] <= t_plane:
            continue
        cos_i = -d[i, 2]
        normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        refracted = eta * d[i] + (eta * cos_i - torch.sqrt(1.0 - eta**2 * (1.0 - cos_i**2))) * normal
        expected[i] = d[i] * t_plane + refracted * (depth_flat[i] - t_plane)

    assert points.shape == (6, 3)
    torch.testing.assert_close(points.double(), expected, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(colors, outputs["rgb"].reshape(-1, 3).clamp(0.0, 1.0))
    assert not torch.allclose(points[2].double(), d[2] * depth_flat[2], atol=1e-3), "oblique water ray must refract"


def test_transform_points_to_bathy_global_frame():
    # NormalizationChain invariant: norm_rot == chunk_rot for georeferenced datasets
    # (see `select_normalization_rotation` in bathyfacto.normalization). The exporter
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


def test_export_bathy_point_cloud_help_lists_the_three_fields(capsys):
    with pytest.raises(SystemExit):
        tyro.cli(ExportBathyPointCloud, args=["--help"])

    help_text = capsys.readouterr().out
    assert "load-config" in help_text
    assert "output-dir" in help_text
    assert "num-points" in help_text


def test_bathyfacto_export_console_script_is_registered():
    console_scripts = importlib.metadata.entry_points(group="console_scripts")
    matching = [ep for ep in console_scripts if ep.name == "bathyfacto-export"]

    assert len(matching) == 1
    assert matching[0].value == "bathyfacto.export:entrypoint"


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


def test_select_bathy_export_depth_prefers_expected_depth_for_bathyfacto():
    outputs = {
        "depth": torch.tensor([[10.0]], dtype=torch.float32),
        "expected_depth": torch.tensor([[2.0]], dtype=torch.float32),
    }

    depth = select_bathy_export_depth(outputs)

    torch.testing.assert_close(depth, outputs["expected_depth"])


def test_no_train_split_warning_follows_the_harmonized_format(recorder):
    """The no-train-split fallback path keeps firing, as a yellow harmonized-prefix line."""

    class FailingDataparser:
        def get_dataparser_outputs(self, split: str):
            raise RuntimeError("no train split available")

    pipeline = SimpleNamespace(datamanager=SimpleNamespace(dataparser=FailingDataparser()))

    mapping = build_export_to_train_index_map(pipeline, SimpleNamespace(image_filenames=[]))

    assert mapping == {}
    assert recorder.messages == [
        "[yellow]bathyfacto  export: no train split available (no train split available), "
        "camera optimizer corrections skipped.[/yellow]"
    ]


def test_success_line_reports_points_written(tmp_path, recorder, monkeypatch):
    """The CLI's final line names the file and the number of points actually written."""
    import open3d as o3d

    fake_pcd = o3d.geometry.PointCloud()
    fake_pcd.points = o3d.utility.Vector3dVector(np.zeros((3, 3), dtype=np.float64))
    fake_pcd.colors = o3d.utility.Vector3dVector(np.zeros((3, 3), dtype=np.float64))

    monkeypatch.setattr(export_module, "eval_setup", lambda load_config: (None, None, None, None))
    monkeypatch.setattr(export_module, "generate_bathy_point_cloud", lambda pipeline, num_points: fake_pcd)

    load_config = tmp_path / "config.yml"
    load_config.write_text("")
    output_dir = tmp_path / "exports" / "bathy-pointcloud"

    ExportBathyPointCloud(load_config=load_config, output_dir=output_dir).main()

    output_path = output_dir / "point_cloud.ply"
    assert output_path.exists()
    assert recorder.messages == [f"bathyfacto  export: 3 points written to {output_path}"]
