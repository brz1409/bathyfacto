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

"""Minimal full-scene refractive point cloud export helpers for Bathy models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.bathy_normalization_utils import NormalizationChain
from nerfstudio.model_components.two_media_geometry import gt_mesh_raycast, snell_refract, water_plane_intersection
from nerfstudio.models.bathyfacto import BathyFactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.math import intersect_aabb
from nerfstudio.utils.rich_utils import CONSOLE

if TYPE_CHECKING:
    import open3d as o3d

UNDERWATER_DEPTH_EPS = 1e-6
# Full-scene Bathy exports retain noticeably more valid land and oblique-view geometry at 1e-2,
# while lower thresholds add very little beyond that on the reference Bathy-Facto run.
ACCUMULATION_THRESHOLD = 0.5


def _resolve_water_entry_epsilon(model: "BathyFactoModel") -> float:
    """Resolve the per-model water-entry epsilon for the exporter.

    ``BathyFactoModel`` keeps the value on ``config.water_entry_epsilon``
    (a concrete ``float``). If an instance attribute ``_water_entry_epsilon``
    is present it takes precedence; otherwise the config value is used.
    """
    resolved = getattr(model, "_water_entry_epsilon", None)
    if resolved is None:
        resolved = model.config.water_entry_epsilon
    if resolved is None:
        raise ValueError(
            f"{type(model).__name__} has no water_entry_epsilon set on the model instance or in its config."
        )
    return float(resolved)


def _get_medium_mask(ray_bundle: RayBundle, *, medium_mask_key: str, threshold: float) -> Tensor:
    metadata = getattr(ray_bundle, "metadata", None)
    if metadata is None or medium_mask_key not in metadata:
        return torch.zeros((len(ray_bundle),), dtype=torch.bool, device=ray_bundle.origins.device)
    mask_tensor = metadata[medium_mask_key]
    if not isinstance(mask_tensor, torch.Tensor):
        mask_tensor = torch.as_tensor(mask_tensor)
    mask_flat = mask_tensor.reshape(-1).to(torch.float32)
    if mask_flat.numel() != len(ray_bundle):
        raise ValueError(f"Medium mask size {mask_flat.numel()} does not match number of rays {len(ray_bundle)}.")
    return (mask_flat > threshold).bool()


def _prepare_flat_geometry_bundle(model: BathyFactoModel, ray_bundle: RayBundle) -> RayBundle:
    geometry_bundle = ray_bundle
    if model.collider is not None:
        geometry_bundle = model.collider(geometry_bundle)
    return geometry_bundle.flatten()


def select_bathy_export_depth(model: BathyFactoModel, outputs: dict[str, Tensor]) -> Tensor:
    """Choose the depth output used for point cloud export."""
    if isinstance(model, BathyFactoModel) and "expected_depth" in outputs:
        return outputs["expected_depth"]
    return outputs["depth"]


def reconstruct_bathy_facto_points_from_depth(
    ray_bundle: RayBundle,
    depth: Tensor,
    *,
    water_plane_normal: Tensor,
    water_plane_d: Tensor,
    scene_box_aabb: Tensor,
    air_refractive_index: float,
    water_refractive_index: float,
    water_entry_epsilon: float,
    water_mask_threshold: float,
    disable_refraction: bool = False,
    gt_water_scene: Optional["o3d.t.geometry.RaycastingScene"] = None,
    gt_water_mesh_t: Optional["o3d.t.geometry.TriangleMesh"] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    origins = ray_bundle.origins
    directions = F.normalize(ray_bundle.directions, dim=-1)
    near_plane = ray_bundle.nears[..., 0]
    far_plane = ray_bundle.fars[..., 0]
    depth_flat = depth.reshape(-1).to(origins.device).to(torch.float32)
    water_pixel_mask = _get_medium_mask(
        ray_bundle,
        medium_mask_key="medium_mask",
        threshold=water_mask_threshold,
    )

    if gt_water_scene is not None and gt_water_mesh_t is not None:
        # PRIOR-03: GT mesh surface — per-ray interface/entry/normals replace the flat plane.
        interface_param, entry_points, local_normal, _ = gt_mesh_raycast(
            origins,
            directions,
            gt_water_scene,
            gt_water_mesh_t,
            water_plane_normal.view(1, 3),
            water_plane_d,
        )
        interface_param = torch.where(torch.isinf(interface_param), far_plane + 1.0, interface_param)
        interface_hits = (interface_param > near_plane) & (interface_param < far_plane) & water_pixel_mask
        refracted_dirs = snell_refract(
            directions,
            local_normal,
            air_refractive_index,
            water_refractive_index,
            disable_refraction=disable_refraction,
        )
        entry_points = origins + directions * interface_param.unsqueeze(-1)
    else:
        interface_param, _ = water_plane_intersection(origins, directions, water_plane_normal.view(1, 3), water_plane_d)
        # Helper returns +inf for parallel rays; clamp to far+1 to preserve downstream mask behavior.
        interface_param = torch.where(torch.isinf(interface_param), far_plane + 1.0, interface_param)
        interface_hits = (interface_param > near_plane) & (interface_param < far_plane) & water_pixel_mask
        refracted_dirs = snell_refract(
            directions,
            water_plane_normal.view(1, 3),
            air_refractive_index,
            water_refractive_index,
            disable_refraction=disable_refraction,
        )
        entry_points = origins + directions * interface_param.unsqueeze(-1)

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
    return points, valid_water_rays, interface_param


def reconstruct_bathy_points(
    model: BathyFactoModel,
    ray_bundle: RayBundle,
    outputs: dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    geometry_bundle = _prepare_flat_geometry_bundle(model, ray_bundle)
    # Normalize directions to physical world units before applying depth / interface parameters.
    geometry_bundle.directions = F.normalize(geometry_bundle.directions, dim=-1)

    depth = select_bathy_export_depth(model, outputs)
    depth_flat = depth.reshape(-1).to(torch.float32)
    accumulation = outputs["accumulation"].reshape(-1).to(torch.float32)
    rgb = outputs["rgb"].reshape(-1, 3).to(torch.float32).clamp(0.0, 1.0)
    points = geometry_bundle.origins + geometry_bundle.directions * depth_flat.unsqueeze(-1)

    plane_normal = model.water_plane_normal.detach().to(dtype=torch.float32, device=geometry_bundle.origins.device)[0]
    plane_d = model.water_plane_d.detach().to(dtype=torch.float32, device=geometry_bundle.origins.device)
    scene_box_aabb = model.scene_box.aabb.detach().to(dtype=torch.float32, device=geometry_bundle.origins.device)
    water_mask_threshold = float(getattr(model.config, "water_mask_threshold", 0.5))

    water_points, valid_water_rays, interface_param = reconstruct_bathy_facto_points_from_depth(
        geometry_bundle,
        depth,
        water_plane_normal=plane_normal,
        water_plane_d=plane_d,
        scene_box_aabb=scene_box_aabb,
        air_refractive_index=model.config.air_refractive_index,
        water_refractive_index=model.config.water_refractive_index,
        water_entry_epsilon=_resolve_water_entry_epsilon(model),
        water_mask_threshold=water_mask_threshold,
        disable_refraction=bool(getattr(model.config, "disable_refraction", False)),
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
    ``chunk_scale := dataparser_outputs.dataparser_scale``), then applies the documented
    chain inline on torch tensors. The NPZ on-disk format is unchanged; this is a
    parameter-object refactor only.
    """
    try:
        chain = NormalizationChain.from_npz_metadata(
            dataparser_outputs.metadata,
            dataparser_scale=float(dataparser_outputs.dataparser_scale),
        )
    except KeyError as exc:
        raise ValueError(f"Bathy point cloud export requires reversible normalization metadata. {exc}") from exc

    norm_rot = torch.as_tensor(chain.norm_rot, dtype=points.dtype, device=points.device)
    norm_center = torch.as_tensor(chain.norm_center, dtype=points.dtype, device=points.device)
    chunk_rot = torch.as_tensor(chain.chunk_rot, dtype=points.dtype, device=points.device)
    chunk_trans = torch.as_tensor(chain.chunk_trans, dtype=points.dtype, device=points.device)

    chunk_local_points = (points * chain.norm_scale + norm_center) @ norm_rot
    global_points = chain.chunk_scale * (chunk_local_points @ chunk_rot.T) + chunk_trans
    return global_points


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
    datamanager = pipeline.datamanager
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


def generate_bathy_point_cloud(
    pipeline: Pipeline, *, num_points: int = 1000000, use_camera_optimizer: bool = True
) -> o3d.geometry.PointCloud:
    if num_points <= 0:
        raise ValueError("num_points must be positive.")

    model = pipeline.model
    if not isinstance(model, BathyFactoModel):
        raise ValueError(f"Bathy point cloud export only supports BathyFactoModel. Got {type(model).__name__}.")
    camera_optimizer = getattr(model, "camera_optimizer", None)

    dataset, dataparser_outputs = get_bathy_export_dataset(pipeline)

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
            metadata = dict(camera_ray_bundle.metadata) if camera_ray_bundle.metadata is not None else {}
            if "medium_mask" in batch:
                metadata["medium_mask"] = batch["medium_mask"].to(dtype=torch.float32, device=model.device)
            camera_ray_bundle.metadata = metadata

            if (
                use_camera_optimizer
                and camera_optimizer is not None
                and camera_optimizer.config.mode != "off"
                and image_idx < camera_optimizer.num_cameras
            ):
                with torch.no_grad():
                    correction = camera_optimizer(torch.tensor([image_idx], device=model.device))  # [1, 3, 4]
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

    points = sampled_points_cpu
    colors = sampled_colors_cpu
    points = transform_points_to_bathy_global_frame(points, dataparser_outputs)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.double().cpu().numpy())
    return pcd
