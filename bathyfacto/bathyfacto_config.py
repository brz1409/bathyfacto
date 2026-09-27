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

"""BathyFacto method and dataparser registrations (Nerfstudio plugin entry points).

The dataparser ``DataParserSpecification`` itself lives in
``bathyfacto.bathyfacto_dataparser_plugin``, not here, and is only re-exported below. See that
module's docstring: ``nerfstudio.configs.dataparser_configs`` discovers the ``bathynerf`` entry
point while ``nerfstudio.pipelines.base_pipeline`` is still partially initialized, so the module
the ``bathynerf`` entry point resolves to must not need ``VanillaPipelineConfig``/``TrainerConfig``
as this module does for the method registrations.
"""

from __future__ import annotations

from typing import Any, Dict

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from bathyfacto.bathyfacto_datamanager import BathyDataManagerConfig
from bathyfacto.bathyfacto_dataparser import BathyNerfDataParserConfig
from bathyfacto.bathyfacto_dataparser_plugin import BathyNerfDataParser
from bathyfacto.bathyfacto_model import BathyFactoModelConfig
from bathyfacto.bathyfacto_pipeline import BathyPipelineConfig

__all__ = ["BathyFactoMethod", "NerfactoBathyCompareMethod", "BathyNerfDataParser"]


def _datamanager() -> BathyDataManagerConfig:
    return BathyDataManagerConfig(
        dataparser=BathyNerfDataParserConfig(),
        train_num_rays_per_batch=4096,
        eval_num_rays_per_batch=4096,
    )


def _optimizers() -> Dict[str, Any]:
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


def _trainer_kwargs() -> Dict[str, Any]:
    return dict(
        steps_per_eval_image=10000,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    )


def _model_kwargs() -> Dict[str, Any]:
    return dict(
        eval_num_rays_per_chunk=1 << 15,
        average_init_density=0.01,
        disable_scene_contraction=True,
        near_plane=0.05,
        far_plane=32.0,
        camera_optimizer=CameraOptimizerConfig(mode="off"),
    )


BathyFactoMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="bathyfacto",
        pipeline=BathyPipelineConfig(
            datamanager=_datamanager(),
            model=BathyFactoModelConfig(**_model_kwargs(), disable_refraction=False),
        ),
        optimizers={
            **_optimizers(),
            # Only used when the camera optimizer is switched on with
            # --pipeline.model.camera-optimizer.mode SO3xR3.
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        **_trainer_kwargs(),
    ),
    description="BathyFacto: two-media NeRF with Snell refraction (paper model, camera poses fixed).",
)

NerfactoBathyCompareMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfacto-bathy-compare",
        pipeline=VanillaPipelineConfig(datamanager=_datamanager(), model=NerfactoModelConfig(**_model_kwargs())),
        optimizers=_optimizers(),
        **_trainer_kwargs(),
    ),
    description="Nerfacto baseline of the BathyFacto paper (single medium, no refraction).",
)
