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

"""Equivalence gate for the paper forward path.

This test checks that the public model keeps the paper forward path.
The frozen reference scalars were captured under identical conditions:
  torch.manual_seed(0), implementation="torch", CPU, 32 downward rays,
  water plane normal=[0,0,1] d=0.5.

Any scalar drift beyond atol=1e-6 signals that the paper code path changed
unexpectedly.

It also asserts that the outputs dict contains exactly the documented keys.
"""

from __future__ import annotations

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox

from bathyfacto.bathyfacto_model import BathyFactoModel, BathyFactoModelConfig

# Frozen reference scalars for the paper configuration (fixed seed, CPU).
_REF_DEPTH_MEAN = 0.86228061
_REF_EXPECTED_DEPTH_MEAN = 0.46407634
_REF_ACCUMULATION_MEAN = 0.52525228
_REF_RGB_MEAN = 0.48588273

# Tight tolerance: the release must reproduce these reference values exactly.
_ATOL = 1e-6
_RTOL = 0


def _make_model() -> BathyFactoModel:
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
    )
    scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    metadata = {"water_surface": {"plane_model": {"normal": [0.0, 0.0, 1.0], "d": 0.5}}}
    return BathyFactoModel(
        config=config,
        scene_box=scene_box,
        num_train_data=4,
        metadata=metadata,
    )


def _make_ray_bundle(model: BathyFactoModel, n_rays: int = 32) -> RayBundle:
    device = next(model.parameters()).device
    origins = torch.zeros(n_rays, 3, device=device)
    directions = torch.zeros(n_rays, 3, device=device)
    directions[:, 2] = -1.0
    return RayBundle(
        origins=origins,
        directions=directions,
        pixel_area=torch.ones(n_rays, 1, device=device) * 1e-5,
        camera_indices=torch.zeros(n_rays, 1, dtype=torch.int32, device=device),
        nears=torch.full((n_rays, 1), 0.05, device=device),
        fars=torch.full((n_rays, 1), 32.0, device=device),
    )


def test_equivalence_gate_scalars_match_frozen_reference() -> None:
    """Model reproduces the 4 frozen reference scalars at atol=1e-6."""
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model)
    outputs = model.get_outputs(rb)

    torch.testing.assert_close(
        outputs["depth"].mean(),
        torch.tensor(_REF_DEPTH_MEAN),
        atol=_ATOL,
        rtol=_RTOL,
    )
    torch.testing.assert_close(
        outputs["expected_depth"].mean(),
        torch.tensor(_REF_EXPECTED_DEPTH_MEAN),
        atol=_ATOL,
        rtol=_RTOL,
    )
    torch.testing.assert_close(
        outputs["accumulation"].mean(),
        torch.tensor(_REF_ACCUMULATION_MEAN),
        atol=_ATOL,
        rtol=_RTOL,
    )
    torch.testing.assert_close(
        outputs["rgb"].mean(),
        torch.tensor(_REF_RGB_MEAN),
        atol=_ATOL,
        rtol=_RTOL,
    )


def test_equivalence_gate_no_nan_inf() -> None:
    """Paper-path outputs contain no NaN or Inf."""
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model)
    outputs = model.get_outputs(rb)

    for key in ("rgb", "depth", "expected_depth", "accumulation"):
        assert not torch.isnan(outputs[key]).any(), f"NaN in {key!r}"
        assert not torch.isinf(outputs[key]).any(), f"Inf in {key!r}"


def test_equivalence_gate_paper_keys_present() -> None:
    """All expected paper-path output keys are present in the outputs dict."""
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model)
    outputs = model.get_outputs(rb)

    for key in ("rgb", "depth", "expected_depth", "accumulation", "interface_hit_mask"):
        assert key in outputs, f"Missing expected paper-path key: {key!r}"


def test_equivalence_gate_output_keys_exact() -> None:
    """The training-mode outputs dict contains exactly the documented keys."""
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model)
    outputs = model.get_outputs(rb)

    expected = {
        "accumulation",
        "depth",
        "distortion_value",
        "expected_depth",
        "interface_hit_mask",
        "interlevel_loss_value",
        "rgb",
    }
    assert set(outputs.keys()) == expected, f"Unexpected output keys: {sorted(outputs.keys())}"
