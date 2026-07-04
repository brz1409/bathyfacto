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

"""Plugin-style Bathy method and dataparser registrations."""

from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.bathynerf_dataparser import BathyNerfDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.bathyfacto import BathyFactoModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification


def _bathy_datamanager_config(
    train_num_rays_per_batch: int = 4096,
    eval_num_rays_per_batch: int = 4096,
) -> VanillaDataManagerConfig:
    return VanillaDataManagerConfig(
        dataparser=BathyNerfDataParserConfig(),
        train_num_rays_per_batch=train_num_rays_per_batch,
        eval_num_rays_per_batch=eval_num_rays_per_batch,
    )


def _compare_optimizers() -> dict:
    """Shared optimizer dict for the comparison configs (no camera_opt)."""
    return {
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    }


# ---------------------------------------------------------------------------
# Paper comparison baselines: Nerfacto (camera optimizer off / SO3xR3)
# ---------------------------------------------------------------------------


def _nerfacto_bathy_compare_config() -> TrainerConfig:
    return TrainerConfig(
        method_name="nerfacto-bathy-compare",
        steps_per_eval_image=10000,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=_bathy_datamanager_config(),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                disable_scene_contraction=True,
                near_plane=0.05,
                far_plane=32.0,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers=_compare_optimizers(),
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    )


def _nerfacto_bathy_compare_camopt_config() -> TrainerConfig:
    return TrainerConfig(
        method_name="nerfacto-bathy-compare-camopt",
        steps_per_eval_image=10000,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=_bathy_datamanager_config(),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                disable_scene_contraction=True,
                near_plane=0.05,
                far_plane=32.0,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
        ),
        optimizers=_compare_optimizers(),
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    )


NerfactoBathyCompareMethod = MethodSpecification(
    config=_nerfacto_bathy_compare_config(),
    description="Nerfacto baseline for bathy comparison (no two-media, no refraction).",
)

NerfactoBathyCompareCamoptMethod = MethodSpecification(
    config=_nerfacto_bathy_compare_camopt_config(),
    description="Nerfacto baseline with camera optimizer SO3xR3 for bathy comparison.",
)


# ---------------------------------------------------------------------------
# BathyFacto: paper model (single-sampler, kinked-ray density, Snell refraction)
# ---------------------------------------------------------------------------


def _bathyfacto_config() -> TrainerConfig:
    return TrainerConfig(
        method_name="bathyfacto",
        steps_per_eval_image=10000,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=_bathy_datamanager_config(),
            model=BathyFactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                disable_scene_contraction=True,
                near_plane=0.05,
                far_plane=32.0,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                disable_refraction=False,
            ),
        ),
        optimizers={
            **_compare_optimizers(),
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    )


BathyFactoMethod = MethodSpecification(
    config=_bathyfacto_config(),
    description="Single-sampler BathyFacto with kinked-ray density wrapper (refraction ON, CamOpt SO3xR3).",
)
