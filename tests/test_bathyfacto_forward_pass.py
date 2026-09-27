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

"""Forward-pass output shapes and the refraction-disabled snapshot for BathyFacto.

The snapshot runs on CPU with the ``torch`` (non-CUDA) implementation at rtol 1e-4, atol 1e-5.
The refraction-enabled reference values live in ``test_bathyfacto_equivalence.py``.
"""

from __future__ import annotations

import torch
from conftest import down_ray_bundle

# --- Tolerance contract ---
RTOL = 1e-4
ATOL = 1e-5


def test_forward_pass_output_shapes(make_tiny_model):
    """get_outputs returns one value per ray for depth and accumulation, three for rgb."""
    torch.manual_seed(0)
    model = make_tiny_model()
    model.train()
    rb = down_ray_bundle(model, 32)
    outputs = model.get_outputs(rb)

    assert outputs["rgb"].shape == (32, 3)
    assert outputs["depth"].shape == (32, 1)
    assert outputs["expected_depth"].shape == (32, 1)
    assert outputs["accumulation"].shape == (32, 1)


def test_forward_pass_snapshot_no_refraction(make_tiny_model):
    """Numeric snapshot: refraction DISABLED, same tolerance."""
    torch.manual_seed(0)
    model = make_tiny_model(disable_refraction=True)
    model.train()
    rb = down_ray_bundle(model, 32)
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
