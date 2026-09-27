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

"""Shared fixtures and helpers: a tiny CPU BathyFacto model, a console recorder, downward rays
and axis rotations.

``test_bathyfacto_equivalence.py`` keeps its own model factory on purpose: its frozen reference
values belong to exactly that configuration, so it must not change with this file.
"""

from typing import Any, Callable, List

import numpy as np
import pytest
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox

from bathyfacto.bathyfacto_model import BathyFactoModel, BathyFactoModelConfig


class RecordingConsole:
    """Stand-in for a module's ``CONSOLE`` that records the raw message text.

    Reading terminal output instead would let rich's highlighting and line wrapping corrupt the
    assertions.
    """

    def __init__(self) -> None:
        self.messages: List[str] = []

    def log(self, message: str, *_args: Any, **_kwargs: Any) -> None:
        self.messages.append(str(message))

    def print(self, message: str, *_args: Any, **_kwargs: Any) -> None:
        self.messages.append(str(message))


@pytest.fixture
def record_console(monkeypatch) -> Callable[[Any], RecordingConsole]:
    """Replace ``module.CONSOLE`` with a :class:`RecordingConsole` and return the recorder."""

    def _record(module: Any) -> RecordingConsole:
        fake = RecordingConsole()
        monkeypatch.setattr(module, "CONSOLE", fake)
        return fake

    return _record


@pytest.fixture
def make_tiny_model() -> Callable[..., BathyFactoModel]:
    """Factory for a small torch-backend BathyFacto model with a water plane at z=-0.5.

    Keyword arguments override fields of ``BathyFactoModelConfig``.
    """

    def _make(**config_overrides: Any) -> BathyFactoModel:
        config = BathyFactoModelConfig(
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
            **config_overrides,
        )
        scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
        metadata = {"water_surface": {"plane_model": {"normal": [0.0, 0.0, 1.0], "d": 0.5}}}
        return BathyFactoModel(config=config, scene_box=scene_box, num_train_data=4, metadata=metadata)

    return _make


def down_ray_bundle(model: BathyFactoModel, n_rays: int, *, all_water: bool = False) -> RayBundle:
    """Straight-down rays from the origin, through the tiny model's water plane at z=-0.5.

    ``all_water`` attaches a medium mask that marks every ray as water.
    """
    device = next(model.parameters()).device
    directions = torch.zeros(n_rays, 3, device=device)
    directions[:, 2] = -1.0
    return RayBundle(
        origins=torch.zeros(n_rays, 3, device=device),
        directions=directions,
        pixel_area=torch.ones(n_rays, 1, device=device) * 1e-5,
        camera_indices=torch.zeros(n_rays, 1, dtype=torch.int32, device=device),
        nears=torch.full((n_rays, 1), 0.05, device=device),
        fars=torch.full((n_rays, 1), 32.0, device=device),
        metadata={"medium_mask": torch.ones(n_rays, 1, device=device)} if all_water else {},
    )


def rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def rotation_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
