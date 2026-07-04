# Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Single-sampler two-media NeRF with kinked-ray density evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.bathy_field import BathyField
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.two_media_geometry import (
    make_kinked_density_fn,
    signed_distance_to_water,
    snell_refract,
    water_plane_intersection,
)
from nerfstudio.models.base_model import Model
from nerfstudio.models.bathy_diagnostics import (
    add_colorbar_to_tensor,
    make_collapse_guard_callback,
    make_depth_monitor_callback,
    metric_to_float,
)
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.math import intersect_aabb


@dataclass
class BathyFactoModelConfig(NerfactoModelConfig):
    """Single-sampler two-media NeRF config — kinked-ray density with medium-conditioned color."""

    _target: Type = field(default_factory=lambda: BathyFactoModel)
    air_refractive_index: float = 1.0
    """Refractive index of air."""
    water_refractive_index: float = 1.333
    """Refractive index of water."""
    water_entry_epsilon: float = 1e-4
    """Offset below water surface before water sampling begins."""
    medium_mask_key: str = "medium_mask"
    """Metadata key used to route rays into the water branch."""
    water_mask_threshold: float = 0.5
    """Threshold applied to the medium mask."""
    valid_mask_key: str = "image_valid_mask"
    """Optional per-pixel validity mask used for losses and metrics."""
    disable_refraction: bool = False
    """Disable refraction at water interface (rays continue straight through)."""
    collapse_guard_enabled: bool = True
    """Stop training early if loss collapses (NaN or sustained spike)."""
    collapse_guard_warmup: int = 1000
    """Steps before collapse detection activates (loss is noisy early on)."""
    collapse_guard_window: int = 100
    """Number of recent losses to keep for baseline median."""
    collapse_guard_spike_factor: float = 5.0
    """Loss must exceed this multiple of the running median to count as a spike."""
    collapse_guard_patience: int = 50
    """Consecutive spike steps before stopping (avoids false positives from single outliers)."""
    depth_monitor_mesh_name: Optional[str] = None
    """Filename of the GT reference mesh (relative to the dataparser data dir). When None, the DepthMonitor
    auto-discovers a `reference_mesh*.ply` in the data dir; set explicitly to override."""


class BathyFactoModel(Model):
    """Single-sampler two-media NeRF with kinked-ray density evaluation."""

    config: BathyFactoModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()

        if self.config.predict_normals:
            raise NotImplementedError("BathyFactoModel does not support predict_normals yet.")

        # Normalize aabb shape
        aabb = self.scene_box.aabb
        if aabb.shape == torch.Size([3, 2]):
            aabb = aabb.T
        if aabb.shape != torch.Size([2, 3]):
            raise ValueError(f"Unexpected scene_box.aabb shape: {tuple(aabb.shape)}")
        self.scene_box.aabb = aabb

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # Single BathyField (medium-conditioned color MLP)
        self.field = BathyField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=False,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
            use_medium_flag=True,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # Proposal networks (shared between air and water — single sampler)
        self.proposal_networks, self.proposal_density_fns = self._build_proposal_networks(scene_contraction)

        def update_schedule(step: int) -> float:
            return float(
                np.clip(
                    np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                    1,
                    self.config.proposal_update_every,
                )
            )

        initial_sampler: Optional[UniformSampler] = None
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        # Single proposal sampler (NOT dual)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.rgb_loss = MSELoss()

        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self._setup_water_interface()

    def _build_proposal_networks(
        self, scene_contraction: Optional[SceneContraction]
    ) -> Tuple[torch.nn.ModuleList, List]:
        """Build proposal density networks — identical to BathyFactoModel."""
        networks = torch.nn.ModuleList()
        density_fns: List = []
        num_prop_nets = self.config.num_proposal_iterations
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            networks.append(network)
            density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
            return networks, density_fns

        for i in range(num_prop_nets):
            prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            networks.append(network)
            density_fns.append(network.density_fn)
        return networks, density_fns

    def _setup_water_interface(self) -> None:
        """Parse water surface plane from metadata and register buffers."""
        metadata = self.kwargs.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("BathyFactoModel requires water surface metadata.")

        water_meta = metadata.get("water_surface")
        if not isinstance(water_meta, Mapping):
            raise ValueError("BathyFactoModel requires metadata['water_surface'].")

        plane_model = water_meta.get("plane_model")
        if not isinstance(plane_model, Mapping):
            raise ValueError("BathyFactoModel requires metadata['water_surface']['plane_model'].")

        normal = plane_model.get("normal")
        d_value = plane_model.get("d")
        if normal is None or d_value is None:
            raise ValueError("BathyFactoModel requires water plane normal and d.")

        normal_tensor = torch.tensor(normal, dtype=torch.float32)
        norm = torch.norm(normal_tensor)
        if norm <= 0:
            raise ValueError("Water surface normal must be non-zero.")
        normal_tensor = normal_tensor / norm
        d = float(d_value) / float(norm)

        self.register_buffer("water_plane_normal", normal_tensor.unsqueeze(0), persistent=False)
        self.register_buffer("water_plane_d", torch.tensor([d], dtype=torch.float32), persistent=False)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            "proposal_networks": list(self.proposal_networks.parameters()),
            "fields": list(self.field.parameters()),
        }
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            n_iters = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step: int) -> None:
                self.step = step
                train_frac = np.clip(step / n_iters, 0, 1)

                def bias(x: float, b: float) -> float:
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(float(train_frac), self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        depth_cb = make_depth_monitor_callback(training_callback_attributes, model=self)
        if depth_cb is not None:
            callbacks.append(depth_cb)
        collapse_cb = make_collapse_guard_callback(
            training_callback_attributes,
            collapse_guard_enabled=self.config.collapse_guard_enabled,
            collapse_guard_warmup=self.config.collapse_guard_warmup,
            collapse_guard_window=self.config.collapse_guard_window,
            collapse_guard_spike_factor=self.config.collapse_guard_spike_factor,
            collapse_guard_patience=self.config.collapse_guard_patience,
        )
        if collapse_cb is not None:
            callbacks.append(collapse_cb)
        return callbacks

    # --- Refraction and geometry helpers ---

    def _signed_distance_to_water(self, positions: Tensor) -> Tensor:
        """Signed distance from positions to water plane. Positive = above water."""
        return signed_distance_to_water(positions, self.water_plane_normal, self.water_plane_d)

    def _compute_refraction(self, incident_dirs: Tensor, n1: float, n2: float, eps: float = 1e-7) -> Tensor:
        """Snell's law refraction with TIR handling. Returns normalized refracted directions."""
        return snell_refract(
            incident_dirs,
            self.water_plane_normal,
            n1,
            n2,
            eps=eps,
            disable_refraction=self.config.disable_refraction,
        )

    def _intersect_scene_box(self, origins: Tensor, directions: Tensor) -> Tuple[Tensor, Tensor]:
        """Intersect rays with scene AABB. Returns (t_near, t_far)."""
        aabb_flat = self.scene_box.aabb.flatten().to(origins.device)
        return intersect_aabb(origins, directions, aabb_flat, invalid_value=0.0)

    def _get_water_pixel_mask(self, ray_bundle: RayBundle, num_rays: int, device: torch.device) -> Optional[Tensor]:
        """Extract water pixel mask from ray_bundle metadata."""
        metadata = getattr(ray_bundle, "metadata", None)
        if metadata is None or self.config.medium_mask_key not in metadata:
            return None
        mask = metadata[self.config.medium_mask_key]
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        mask_flat = mask.to(device=device).reshape(-1).to(torch.float32)
        if mask_flat.numel() != num_rays:
            raise ValueError(f"Medium mask size {mask_flat.numel()} does not match number of rays {num_rays}.")
        return mask_flat > float(self.config.water_mask_threshold)

    # --- Forward pass ---

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:
        """Single-sampler two-media rendering pipeline.

        Constructs a virtual straight ray spanning air + water, runs proposal sampling
        with a kinked density function (Snell-corrected water positions), then renders
        RGB/depth/accumulation with medium-conditioned color.

        Returns dict with keys: rgb, accumulation, depth, expected_depth,
        interlevel_loss_value, distortion_value.
        """
        # 1. Camera optimizer
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        # 2. Flatten and extract basics
        ray_bundle = ray_bundle.flatten()
        num_rays = len(ray_bundle)
        device = ray_bundle.origins.device

        origins = ray_bundle.origins
        directions = F.normalize(ray_bundle.directions, dim=-1)
        near_plane = ray_bundle.nears[..., 0] if ray_bundle.nears is not None else torch.zeros(num_rays, device=device)
        far_plane = (
            ray_bundle.fars[..., 0]
            if ray_bundle.fars is not None
            else torch.full((num_rays,), self.config.far_plane, device=device)
        )

        # 3. Water interface intersection
        interface_param, _ = water_plane_intersection(origins, directions, self.water_plane_normal, self.water_plane_d)
        # Helper returns +inf for parallel rays; clamp to far+1 to preserve downstream mask behavior.
        interface_param = torch.where(torch.isinf(interface_param), far_plane + 1.0, interface_param)

        # 4. Interface hits (with water pixel mask)
        interface_hits = (interface_param > near_plane) & (interface_param < far_plane)
        water_pixel_mask = self._get_water_pixel_mask(ray_bundle, num_rays, device=device)
        if water_pixel_mask is not None:
            interface_hits = interface_hits & water_pixel_mask

        # 5. Water-plane entry points
        entry_points = origins + directions * interface_param.unsqueeze(-1)

        # 6. Snell refraction
        refracted_dirs = self._compute_refraction(
            directions, self.config.air_refractive_index, self.config.water_refractive_index
        )

        # 7. Water far bounds
        _, t_far_water = self._intersect_scene_box(entry_points, refracted_dirs)
        t_far_water = t_far_water.clamp(min=self.config.water_entry_epsilon)

        # 8. Total far: interface_param + t_far_water for water rays, far_plane for air-only
        total_far = torch.where(interface_hits, interface_param + t_far_water, far_plane)
        # Guarantee total_far >= near + eps so the RayBundle never degenerates.
        total_far = torch.maximum(total_far, near_plane + self.config.water_entry_epsilon)

        # 9. Virtual RayBundle with extended far
        virtual_rb = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=ray_bundle.pixel_area,
            camera_indices=ray_bundle.camera_indices,
            nears=near_plane.unsqueeze(-1),
            fars=total_far.unsqueeze(-1),
            metadata=ray_bundle.metadata,
            times=ray_bundle.times,
        )

        # 10. Proposal sampling with kinked density
        kinked_fns = [
            make_kinked_density_fn(fn, origins, directions, interface_param, entry_points, refracted_dirs)
            for fn in self.proposal_density_fns
        ]
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(virtual_rb, density_fns=kinked_fns)

        # 11. Correct final sample positions for the kink
        starts = ray_samples.frustums.starts  # [N, S, 1]
        ends = ray_samples.frustums.ends  # [N, S, 1]
        t_mid = (starts + ends) / 2  # [N, S, 1]
        ip = interface_param.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
        is_water = (t_mid > ip) & interface_hits.unsqueeze(-1).unsqueeze(-1)  # [N, S, 1]

        water_t = t_mid - ip
        water_pos = entry_points.unsqueeze(1) + refracted_dirs.unsqueeze(1) * water_t  # [N, S, 3]
        air_pos = origins.unsqueeze(1) + directions.unsqueeze(1) * t_mid  # [N, S, 3]
        corrected_pos = torch.where(is_water, water_pos, air_pos)

        # Corrected directions per sample
        corrected_dirs = torch.where(
            is_water.expand_as(air_pos),
            refracted_dirs.unsqueeze(1).expand_as(air_pos),
            directions.unsqueeze(1).expand_as(air_pos),
        )

        # 12. Build corrected RaySamples for field evaluation
        corrected_samples = RaySamples(
            frustums=Frustums(
                origins=corrected_pos,
                directions=corrected_dirs,
                starts=torch.zeros_like(starts),
                ends=torch.zeros_like(ends),
                pixel_area=ray_samples.frustums.pixel_area,
            ),
            camera_indices=ray_samples.camera_indices,
            deltas=ray_samples.deltas,
            spacing_starts=ray_samples.spacing_starts,
            spacing_ends=ray_samples.spacing_ends,
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
            metadata=ray_samples.metadata,
            times=ray_samples.times,
        )

        # 13. Evaluate field
        density, base_mlp_out = self.field.get_density(corrected_samples)
        is_water_flag = is_water.squeeze(-1).unsqueeze(-1).float()  # [N, S, 1]
        field_outputs = self.field.get_outputs(
            corrected_samples, density_embedding=base_mlp_out, medium_flag=is_water_flag
        )

        # 14. Weights from ORIGINAL ray_samples (t-value based)
        weights = ray_samples.get_weights(density)

        # 15. Render
        rgb_samples = field_outputs[FieldHeadNames.RGB]
        rgb = self.renderer_rgb(rgb=rgb_samples, weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights)

        outputs: Dict[str, Tensor] = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "interface_hit_mask": interface_hits.unsqueeze(-1),
        }

        # 16. Training losses
        if self.training:
            all_weights_list = weights_list + [weights]
            all_samples_list = ray_samples_list + [ray_samples]
            outputs["interlevel_loss_value"] = interlevel_loss(all_weights_list, all_samples_list)
            outputs["distortion_value"] = distortion_loss(all_weights_list, all_samples_list)

        return outputs

    # --- Loss, metrics, evaluation ---

    def _get_valid_mask(self, batch: Dict[str, Tensor], device: torch.device) -> Optional[Tensor]:
        if self.config.valid_mask_key not in batch:
            return None
        valid_mask = batch[self.config.valid_mask_key].to(device)
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = torch.as_tensor(valid_mask, device=device)
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()
        if valid_mask.ndim >= 2 and valid_mask.shape[-1] == 1:
            valid_mask = valid_mask.squeeze(-1)
        return valid_mask

    def get_loss_dict(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor], metrics_dict=None):
        del metrics_dict
        loss_dict: Dict[str, Tensor] = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        valid_mask = self._get_valid_mask(batch, self.device)
        if valid_mask is not None:
            valid_mask_rgb = valid_mask.unsqueeze(-1).expand_as(pred_rgb)
            pred_rgb = torch.where(valid_mask_rgb, pred_rgb, torch.zeros_like(pred_rgb))
            gt_rgb = torch.where(valid_mask_rgb, gt_rgb, torch.zeros_like(gt_rgb))
            num_valid = valid_mask.sum().clamp(min=1.0)
            loss_scale = valid_mask.numel() / num_valid
            loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb) * loss_scale
        else:
            loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * outputs["interlevel_loss_value"]
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * outputs["distortion_value"]
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    def get_metrics_dict(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor], step: Optional[int] = None):
        del step
        metrics_dict: Dict[str, Any] = {}
        gt_rgb = batch["image"].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        predicted_rgb = outputs["rgb"]
        predicted_rgb_is_finite = bool(torch.isfinite(predicted_rgb).all())
        valid_mask = self._get_valid_mask(batch, self.device)
        if not predicted_rgb_is_finite:
            metrics_dict["psnr"] = torch.zeros((), device=self.device)
        elif valid_mask is not None and valid_mask.any():
            mask_flat = valid_mask.reshape(-1)
            gt_flat = gt_rgb.reshape(-1, 3)
            pred_flat = predicted_rgb.reshape(-1, 3)
            mse = ((pred_flat[mask_flat] - gt_flat[mask_flat]) ** 2).mean()
            metrics_dict["psnr"] = -10.0 * torch.log10(mse.clamp(min=1e-10))
        else:
            metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training and "distortion_value" in outputs:
            metrics_dict["distortion"] = outputs["distortion_value"]

        self.camera_optimizer.get_metrics_dict(metrics_dict)

        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor], step: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        del step
        gt_rgb = batch["image"].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        predicted_rgb = outputs["rgb"]
        valid_mask = self._get_valid_mask(batch, self.device)

        def apply_invalid_overlay(image: Tensor) -> Tensor:
            if valid_mask is None:
                return image
            mask_rgb = valid_mask.unsqueeze(-1).expand_as(image)
            return torch.where(mask_rgb, image, torch.zeros_like(image))

        display_gt = apply_invalid_overlay(gt_rgb)
        display_pred = apply_invalid_overlay(predicted_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([display_gt, display_pred], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        combined_acc = add_colorbar_to_tensor(combined_acc, title="Accumulation (Opacity)", min_val=0.0, max_val=1.0)
        valid_depths = outputs["depth"][outputs["depth"] > 0]
        if valid_depths.numel() > 0:
            depth_min = valid_depths.min().item()
            depth_max = valid_depths.max().item()
        else:
            depth_min, depth_max = 0.0, 1.0

        combined_depth = add_colorbar_to_tensor(
            combined_depth, title="Depth (Meters)", min_val=depth_min, max_val=depth_max
        )

        if valid_mask is not None and valid_mask.any():
            mask_flat = valid_mask.reshape(-1)
            gt_flat = gt_rgb.reshape(-1, 3)
            pred_flat = predicted_rgb.reshape(-1, 3)
            mse = ((pred_flat[mask_flat] - gt_flat[mask_flat]) ** 2).mean()
            psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))

            gt_eval = apply_invalid_overlay(gt_rgb)
            pred_eval = apply_invalid_overlay(predicted_rgb)
            gt_eval = torch.moveaxis(gt_eval, -1, 0)[None, ...]
            pred_eval = torch.moveaxis(pred_eval, -1, 0)[None, ...]
        else:
            gt_eval = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
            pred_eval = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]
            psnr = self.psnr(gt_eval, pred_eval)

        # Clamp to [0, 1] for perceptual metrics (HDR composites can push pixels above 1.0)
        gt_eval = gt_eval.clamp(0.0, 1.0)
        pred_eval = pred_eval.clamp(0.0, 1.0)
        ssim = self.ssim(gt_eval, pred_eval)
        lpips = self.lpips(gt_eval, pred_eval)

        metrics_dict = {
            "psnr": metric_to_float(psnr),
            "ssim": metric_to_float(ssim),
            "lpips": metric_to_float(lpips),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera(
        self,
        camera: Cameras,
        obb_box=None,
        batch: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        medium_mask = None
        if batch is not None and self.config.medium_mask_key in batch:
            medium_mask = batch[self.config.medium_mask_key]
        elif camera.metadata is not None and self.config.medium_mask_key in camera.metadata:
            medium_mask = camera.metadata[self.config.medium_mask_key]

        camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
        if medium_mask is not None:
            if not isinstance(medium_mask, torch.Tensor):
                medium_mask = torch.as_tensor(medium_mask)
            medium_mask = medium_mask.detach().to(camera_ray_bundle.origins.device)
            if medium_mask.numel() != len(camera_ray_bundle):
                raise ValueError(
                    f"Medium mask size {medium_mask.numel()} does not match number of rays {len(camera_ray_bundle)}."
                )
            spatial_shape = camera_ray_bundle.origins.shape[:-1]
            metadata = dict(camera_ray_bundle.metadata) if camera_ray_bundle.metadata is not None else {}
            metadata[self.config.medium_mask_key] = medium_mask.reshape(*spatial_shape, 1).to(torch.float32)
            camera_ray_bundle.metadata = metadata

        return self.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

    def get_rgba_image(self, outputs: Dict[str, Tensor], output_name: str = "rgb") -> Tensor:
        accumulation_name = output_name.replace("rgb", "accumulation")
        rgb = outputs[output_name]
        acc = outputs[accumulation_name]
        if acc.dim() < rgb.dim():
            acc = acc.unsqueeze(-1)
        return torch.cat((rgb, acc), dim=-1)
