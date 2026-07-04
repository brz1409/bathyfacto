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

"""Forward-pass snapshot gate for BathyFacto.

Per-commit numeric gate: every commit must pass this snapshot test.

Tolerance contract (relaxed from bit-identity):

    rtol = 1e-4
    atol = 1e-5

Snapshot fixtures captured against the BathyFacto single-sampler pipeline
on CPU with the ``torch`` (non-CUDA) implementation. The values are reproducible
to ~12 digits on the same code path, so the bounds above leave a generous safety
margin for floating-point drift introduced by cleanups.

If a future commit legitimately changes any of these values, regenerate the
snapshot in the SAME commit and call it out explicitly in the commit message.
"""

from __future__ import annotations

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.bathyfacto import BathyFactoModel, BathyFactoModelConfig

# --- Tolerance contract ---
RTOL = 1e-4
ATOL = 1e-5


def _make_model(disable_refraction: bool = False, num_train_data: int = 4) -> BathyFactoModel:
    config = BathyFactoModelConfig(
        disable_refraction=disable_refraction,
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
        num_train_data=num_train_data,
        metadata=metadata,
    )


def _make_ray_bundle(model: BathyFactoModel, n_rays: int = 32) -> RayBundle:
    """Downward-pointing rays at origin (z=0); water plane at z=-0.5 inside SceneBox."""
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


def test_forward_pass_keys_and_shapes():
    """get_outputs returns the documented keys with expected shapes and no NaN/Inf."""
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model, n_rays=32)
    outputs = model.get_outputs(rb)

    for key in ("rgb", "depth", "expected_depth", "accumulation"):
        assert key in outputs, f"missing output key: {key!r}"

    assert outputs["rgb"].shape == (32, 3)
    assert outputs["depth"].shape == (32, 1)
    assert outputs["expected_depth"].shape == (32, 1)
    assert outputs["accumulation"].shape == (32, 1)

    for key in ("rgb", "depth", "expected_depth", "accumulation"):
        assert not torch.isnan(outputs[key]).any(), f"NaN in {key!r}"
        assert not torch.isinf(outputs[key]).any(), f"Inf in {key!r}"


def test_forward_pass_snapshot_refraction_on():
    """Numeric snapshot — refraction ON, tolerance rtol=1e-4, atol=1e-5."""
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    rb = _make_ray_bundle(model, n_rays=32)
    outputs = model.get_outputs(rb)

    torch.testing.assert_close(
        outputs["depth"].mean(),
        torch.tensor(0.86228061),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        outputs["expected_depth"].mean(),
        torch.tensor(0.46407634),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        outputs["accumulation"].mean(),
        torch.tensor(0.52525228),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        outputs["rgb"].mean(),
        torch.tensor(0.48588273),
        rtol=RTOL,
        atol=ATOL,
    )


def test_forward_pass_snapshot_no_refraction():
    """Numeric snapshot — refraction DISABLED, same tolerance."""
    torch.manual_seed(0)
    model = _make_model(disable_refraction=True)
    model.train()
    rb = _make_ray_bundle(model, n_rays=32)
    outputs = model.get_outputs(rb)

    torch.testing.assert_close(
        outputs["depth"].mean(),
        torch.tensor(0.86228061),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        outputs["accumulation"].mean(),
        torch.tensor(0.52525228),
        rtol=RTOL,
        atol=ATOL,
    )
