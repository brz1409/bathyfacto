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

"""Minimal full-scene refractive point cloud export helpers for Bathy models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast

import torch
import torch.nn.functional as F
import tyro
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.math import intersect_aabb
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import Tensor

from bathyfacto.bathyfacto_datamanager import MEDIUM_MASK_KEY, attach_medium_mask
from bathyfacto.bathyfacto_model import BathyFactoModel
from bathyfacto.normalization import NormalizationChain

if TYPE_CHECKING:
    import open3d as o3d

UNDERWATER_DEPTH_EPS = 1e-6
# Rays below this accumulation are background and are dropped from the export. The published
# cloud-to-mesh numbers were measured with 0.5.
ACCUMULATION_THRESHOLD = 0.5


def select_bathy_export_depth(outputs: dict[str, Tensor]) -> Tensor:
    """Choose the depth output used for point cloud export."""
    if "expected_depth" in outputs:
        return outputs["expected_depth"]
    return outputs["depth"]


def reconstruct_bathy_facto_points_from_depth(
    depth: Tensor,
    *,
    interface_param: Tensor,
    entry_points: Tensor,
    refracted_dirs: Tensor,
    interface_hits: Tensor,
    scene_box_aabb: Tensor,
    water_entry_epsilon: float,
) -> Tuple[Tensor, Tensor]:
    """Place each ray's rendered depth on its refracted water segment.

    The geometry (``interface_param``, ``entry_points``, ``refracted_dirs``, ``interface_hits``)
    is the one the model published for this forward pass, flattened to ``(N,)`` / ``(N, 3)``.

    Returns:
        The refracted points ``(N, 3)`` and the rays whose water segment is valid ``(N,)``.
    """
    depth_flat = depth.reshape(-1).to(entry_points.device).to(torch.float32)
    t_near_water, t_far_water = intersect_aabb(
        entry_points,
        refracted_dirs,
        scene_box_aabb.flatten(),
        invalid_value=0.0,
    )
    water_nears = torch.maximum(t_near_water, torch.full_like(t_near_water, float(water_entry_epsilon)))
    valid_water_rays = interface_hits & (t_far_water > water_nears)

    water_distance = (depth_flat - interface_param).clamp_min(0.0)
    points = entry_points + refracted_dirs * water_distance.unsqueeze(-1)
    return points, valid_water_rays


def reconstruct_bathy_points(
    model: BathyFactoModel,
    ray_bundle: RayBundle,
    outputs: dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    geometry_bundle = ray_bundle.flatten()
    # Unit directions, so depth and interface_param are distances along the ray in scene units.
    directions = F.normalize(geometry_bundle.directions, dim=-1)

    depth = select_bathy_export_depth(outputs)
    depth_flat = depth.reshape(-1).to(torch.float32)
    accumulation = outputs["accumulation"].reshape(-1).to(torch.float32)
    rgb = outputs["rgb"].reshape(-1, 3).to(torch.float32).clamp(0.0, 1.0)
    points = geometry_bundle.origins + directions * depth_flat.unsqueeze(-1)

    interface_param = outputs["interface_param"].reshape(-1)
    water_points, valid_water_rays = reconstruct_bathy_facto_points_from_depth(
        depth,
        interface_param=interface_param,
        entry_points=outputs["entry_points"].reshape(-1, 3),
        refracted_dirs=outputs["refracted_dirs"].reshape(-1, 3),
        interface_hits=outputs["interface_hit_mask"].reshape(-1),
        scene_box_aabb=model.scene_box.aabb.detach().to(dtype=torch.float32, device=points.device),
        water_entry_epsilon=model.config.water_entry_epsilon,
    )

    underwater_hits = valid_water_rays & (depth_flat > (interface_param + UNDERWATER_DEPTH_EPS))
    points[underwater_hits] = water_points[underwater_hits]

    export_mask = accumulation > ACCUMULATION_THRESHOLD
    export_mask &= torch.isfinite(points).all(dim=-1)
    export_mask &= torch.isfinite(rgb).all(dim=-1)
    export_mask &= torch.isfinite(depth_flat)
    export_mask &= depth_flat > 0

    return points[export_mask], rgb[export_mask]


def transform_points_to_bathy_global_frame(points: Tensor, dataparser_outputs: DataparserOutputs) -> Tensor:
    """Apply the scene → global transform to ``points`` using ``dataparser_outputs.metadata``.

    Builds a :class:`NormalizationChain` from the dataparser's metadata (with
    ``chunk_scale := dataparser_outputs.dataparser_scale``) and applies it.
    """
    try:
        chain = NormalizationChain.from_npz_metadata(
            dataparser_outputs.metadata,
            dataparser_scale=float(dataparser_outputs.dataparser_scale),
        )
    except KeyError as exc:
        raise ValueError(f"Bathy point cloud export requires reversible normalization metadata. {exc}") from exc
    return chain.scene_to_global(points)


def update_bathy_point_reservoir(
    reservoir_points: Optional[Tensor],
    reservoir_colors: Optional[Tensor],
    reservoir_keys: Optional[Tensor],
    candidate_points: Tensor,
    candidate_colors: Tensor,
    *,
    num_points: int,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor, Tensor]:
    if candidate_points.shape[0] != candidate_colors.shape[0]:
        raise ValueError("Candidate point and color counts must match.")
    if num_points <= 0:
        raise ValueError("num_points must be positive.")

    candidate_keys = torch.rand(candidate_points.shape[0], generator=generator, dtype=torch.float32)
    if candidate_points.shape[0] > num_points:
        _, keep_indices = torch.topk(candidate_keys, k=num_points, largest=False)
        candidate_points = candidate_points[keep_indices]
        candidate_colors = candidate_colors[keep_indices]
        candidate_keys = candidate_keys[keep_indices]

    if reservoir_points is None or reservoir_colors is None or reservoir_keys is None:
        combined_points = candidate_points
        combined_colors = candidate_colors
        combined_keys = candidate_keys
    else:
        combined_points = torch.cat([reservoir_points, candidate_points], dim=0)
        combined_colors = torch.cat([reservoir_colors, candidate_colors], dim=0)
        combined_keys = torch.cat([reservoir_keys, candidate_keys], dim=0)

    if combined_points.shape[0] > num_points:
        _, keep_indices = torch.topk(combined_keys, k=num_points, largest=False)
        combined_points = combined_points[keep_indices]
        combined_colors = combined_colors[keep_indices]
        combined_keys = combined_keys[keep_indices]

    return combined_points, combined_colors, combined_keys


def get_bathy_export_dataset(pipeline: Pipeline) -> Tuple[InputDataset, DataparserOutputs]:
    datamanager = cast(Any, pipeline.datamanager)
    if not hasattr(datamanager, "dataparser"):
        raise ValueError("Bathy point cloud export requires a dataparser-backed pipeline.")

    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split="data")
    dataset_type = (
        type(datamanager.train_dataset) if getattr(datamanager, "train_dataset", None) is not None else InputDataset
    )
    camera_res_scale_factor = getattr(getattr(datamanager, "config", None), "camera_res_scale_factor", 1.0)
    export_dataset = dataset_type(
        dataparser_outputs=dataparser_outputs,
        scale_factor=camera_res_scale_factor,
    )
    return export_dataset, dataparser_outputs


def build_export_to_train_index_map(pipeline: Pipeline, dataparser_outputs: DataparserOutputs) -> dict[int, int]:
    """Map each export-dataset image index to its index in the TRAIN split, by filename.

    The export dataset is the ``"data"`` split, which holds every image in the scene. The camera
    optimizer, however, only ever saw the ``"train"`` split and is indexed by TRAIN position.
    Feeding it an export index therefore reads out some other camera's correction: for a
    130-image / 117-train dataset, every index past the first held-out image is off, and the
    last 13 are out of range entirely.

    Returns a dict keyed by export index. Images absent from the train split (the val/test
    held-out cameras) are missing from the map. They have no learned correction, and the
    caller must leave their poses untouched instead of substituting a neighbour's.
    """
    datamanager = cast(Any, pipeline.datamanager)
    try:
        train_outputs = datamanager.dataparser.get_dataparser_outputs(split="train")
    except Exception as err:  # dataparser may not expose a train split at export time
        CONSOLE.log(
            f"[yellow]bathyfacto  export: no train split available ({err}), "
            "camera optimizer corrections skipped.[/yellow]"
        )
        return {}

    train_index_by_name = {Path(name).name: idx for idx, name in enumerate(train_outputs.image_filenames)}
    return {
        export_idx: train_index_by_name[Path(name).name]
        for export_idx, name in enumerate(dataparser_outputs.image_filenames)
        if Path(name).name in train_index_by_name
    }


def generate_bathy_point_cloud(
    pipeline: Pipeline, *, num_points: int = 1000000, use_camera_optimizer: bool = True
) -> o3d.geometry.PointCloud:
    """Export the refracted point cloud, rendered with the pipeline in eval mode.

    The pipeline is switched to eval mode for the export and every module gets its previous
    train/eval flag back afterwards, also when the export raises. ``ns-eval``-style loading
    already hands over an eval-mode pipeline, so the CLI output does not depend on this.
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive.")

    previous_modes = [(module, module.training) for module in pipeline.modules()]
    pipeline.eval()
    try:
        return _generate_bathy_point_cloud(pipeline, num_points=num_points, use_camera_optimizer=use_camera_optimizer)
    finally:
        for module, was_training in previous_modes:
            module.training = was_training


def _generate_bathy_point_cloud(
    pipeline: Pipeline, *, num_points: int, use_camera_optimizer: bool
) -> o3d.geometry.PointCloud:
    model = pipeline.model
    if not isinstance(model, BathyFactoModel):
        raise ValueError(f"Bathy point cloud export only supports BathyFactoModel. Got {type(model).__name__}.")
    camera_optimizer = model.camera_optimizer
    apply_camera_optimizer = use_camera_optimizer and camera_optimizer.config.mode != "off"

    dataset, dataparser_outputs = get_bathy_export_dataset(pipeline)

    export_to_train_idx: dict[int, int] = {}
    if apply_camera_optimizer:
        export_to_train_idx = build_export_to_train_index_map(pipeline, dataparser_outputs)
        n_corrected, n_total = len(export_to_train_idx), len(dataparser_outputs.image_filenames)
        CONSOLE.log(
            f"bathyfacto  export: camera optimizer corrections for {n_corrected} train cameras, "
            f"{n_total - n_corrected} held-out cameras keep their original pose"
        )

    import open3d as o3d

    sampled_points_cpu: Optional[Tensor] = None
    sampled_colors_cpu: Optional[Tensor] = None
    sampled_keys_cpu: Optional[Tensor] = None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    progress = Progress(
        TextColumn(":ocean: Exporting Bathy point cloud :ocean:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )
    with progress:
        task = progress.add_task("Exporting Bathy point cloud", total=len(dataset))
        for image_idx in range(len(dataset)):
            batch = dataset[image_idx]

            camera_ray_bundle = dataset.cameras.generate_rays(camera_indices=image_idx, keep_shape=True).to(
                model.device
            )
            attach_medium_mask(camera_ray_bundle, batch.get(MEDIUM_MASK_KEY))

            # The camera optimizer is indexed by TRAIN position, not by position in the
            # export ("data") split. Held-out cameras have no learned correction and are
            # exported with their original pose.
            train_idx = export_to_train_idx.get(image_idx)
            if apply_camera_optimizer and train_idx is not None and train_idx < camera_optimizer.num_cameras:
                with torch.no_grad():
                    correction = camera_optimizer(torch.tensor([train_idx], device=model.device))  # [1, 3, 4]
                camera_ray_bundle.origins = camera_ray_bundle.origins + correction[0, :3, 3]
                camera_ray_bundle.directions = F.normalize(
                    (correction[0, :3, :3] @ camera_ray_bundle.directions.unsqueeze(-1)).squeeze(-1), dim=-1
                )

            with torch.no_grad():
                outputs = model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            image_points, image_colors = reconstruct_bathy_points(model, camera_ray_bundle, outputs)
            if image_points.numel() > 0:
                sampled_points_cpu, sampled_colors_cpu, sampled_keys_cpu = update_bathy_point_reservoir(
                    sampled_points_cpu,
                    sampled_colors_cpu,
                    sampled_keys_cpu,
                    image_points.cpu(),
                    image_colors.cpu(),
                    num_points=num_points,
                    generator=generator,
                )
            progress.advance(task, 1)

    if sampled_points_cpu is None or sampled_colors_cpu is None:
        raise ValueError("Bathy point cloud export produced no valid refracted points.")

    points = transform_points_to_bathy_global_frame(sampled_points_cpu, dataparser_outputs)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(sampled_colors_cpu.double().cpu().numpy())
    return pcd


def resolve_bathy_output_dir(load_config: Path, output_dir: Optional[Path]) -> Path:
    """Default output directory: <run_dir>/exports/bathy-pointcloud."""
    if output_dir is not None:
        return output_dir
    return load_config.parent / "exports" / "bathy-pointcloud"


@dataclass
class ExportBathyPointCloud:
    """Export the refraction-corrected BathyFacto point cloud in the global frame of the dataset."""

    load_config: Path
    """Path to the config.yml of the trained run."""
    output_dir: Optional[Path] = None
    """Output directory. Defaults to <run_dir>/exports/bathy-pointcloud."""
    num_points: int = 1000000
    """Maximum number of exported points after filtering."""

    def main(self) -> None:
        import open3d as o3d

        output_dir = resolve_bathy_output_dir(self.load_config, self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _, pipeline, _, _ = eval_setup(self.load_config)
        pcd = generate_bathy_point_cloud(pipeline, num_points=self.num_points)
        torch.cuda.empty_cache()
        num_written = len(pcd.points)
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        output_path = output_dir / "point_cloud.ply"
        o3d.t.io.write_point_cloud(str(output_path), tpcd)
        CONSOLE.print(f"bathyfacto  export: {num_written} points written to {output_path}")


def entrypoint() -> None:
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ExportBathyPointCloud).main()


if __name__ == "__main__":
    entrypoint()
