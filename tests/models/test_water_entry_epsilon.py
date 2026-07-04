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

"""Verify water_entry_epsilon handling in BathyFactoModel and the exporter.

The exporter ``bathy_pointcloud_utils`` uses:

    water_entry_epsilon = float(getattr(model, "_water_entry_epsilon", 1e-4))

For ``BathyFactoModel`` the attribute ``_water_entry_epsilon`` is *never*
set on the model instance. ``getattr`` falls through to the hard-coded fallback
``1e-4``, which coincides with the dataclass default of
``BathyFactoModelConfig.water_entry_epsilon``. For any user-customised
value the exporter silently uses ``1e-4`` — a semantic inconsistency for
BathyFacto. These tests document and pin that invariant.
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from nerfstudio.models.bathyfacto import BathyFactoModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bathyfacto_config_has_default_water_entry_epsilon_1e_minus_4():
    """``BathyFactoModelConfig.water_entry_epsilon`` default is ``1e-4``.

    The exporter's hard-coded fallback ``1e-4`` is only safe because of this
    default. Any change to the default would silently desync exporter behaviour
    from training behaviour for BathyFacto.
    """
    fields = {f.name: f for f in dataclasses.fields(BathyFactoModelConfig)}
    assert "water_entry_epsilon" in fields, (
        "BathyFactoModelConfig must declare water_entry_epsilon (exporter call site depends on it)."
    )
    default = fields["water_entry_epsilon"].default
    assert default == 1e-4, (
        f"Default water_entry_epsilon is {default!r}; exporter fallback 1e-4 would silently disagree."
    )


def test_bathyfacto_model_has_no_water_entry_epsilon_attribute():
    """``BathyFactoModel`` does NOT set ``self._water_entry_epsilon``.

    getattr falls through to the hard-coded 1e-4 fallback for every
    BathyFacto export, regardless of what the user configured.
    """
    pytest.importorskip("torch")
    import torch

    from nerfstudio.data.scene_box import SceneBox
    from nerfstudio.models.bathyfacto import BathyFactoModel

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
        water_entry_epsilon=5e-3,  # explicitly different from the 1e-4 fallback
    )
    scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    metadata = {"water_surface": {"plane_model": {"normal": [0.0, 0.0, 1.0], "d": 0.5}}}
    model = BathyFactoModel(config=config, scene_box=scene_box, num_train_data=4, metadata=metadata)

    assert not hasattr(model, "_water_entry_epsilon"), (
        "BathyFacto must not set _water_entry_epsilon; if that changed the verify rule needs to be re-validated."
    )
    # config still holds the user value
    assert model.config.water_entry_epsilon == 5e-3
    # exporter's current getattr fallback would silently drop to 1e-4 for this model:
    silent_fallback = float(getattr(model, "_water_entry_epsilon", 1e-4))
    assert silent_fallback == 1e-4
    assert silent_fallback != model.config.water_entry_epsilon, (
        "If these ever coincide for a non-default config, the bug becomes invisible by accident."
    )


def test_no_production_writer_for_underscore_water_entry_epsilon_on_bathyfacto():
    """Grep confirms no production code writes ``model._water_entry_epsilon`` on BathyFacto.

    Allowed hits: exporter reader (getattr) and config dataclass field definition.
    Disallowed: any writer inside bathyfacto.py or any other BathyFacto-related module.
    """
    result = subprocess.run(
        ["grep", "-rn", "_water_entry_epsilon", "nerfstudio/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    hits = [line for line in result.stdout.splitlines() if line.strip()]
    bathyfacto_writers = [
        line
        for line in hits
        if "bathyfacto.py" in line and "_water_entry_epsilon" in line and "self._water_entry_epsilon" in line
    ]
    assert bathyfacto_writers == [], (
        f"BathyFacto writes _water_entry_epsilon; invariant broken:\n{bathyfacto_writers}"
    )
