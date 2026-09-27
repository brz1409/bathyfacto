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

"""BathyPipeline passes the eval batch (and with it the medium mask) to the model."""

from unittest.mock import MagicMock

import torch

from bathyfacto.bathyfacto_pipeline import BathyPipeline


def _pipeline_with(camera, batch):
    pipe = BathyPipeline.__new__(BathyPipeline)
    torch.nn.Module.__init__(pipe)
    pipe.datamanager = MagicMock()
    pipe.datamanager.next_eval_image.return_value = (camera, batch)
    pipe._model = MagicMock()
    pipe._model.get_image_metrics_and_images.return_value = ({"psnr": torch.tensor(1.0)}, {})
    return pipe


def test_single_image_eval_passes_batch():
    camera = MagicMock(height=torch.tensor(2), width=torch.tensor(3), size=1)
    batch = {"medium_mask": torch.ones(2, 3, 1, dtype=torch.bool)}
    pipe = _pipeline_with(camera, batch)
    pipe.get_eval_image_metrics_and_images(step=0)
    _, kwargs = pipe._model.get_outputs_for_camera.call_args
    assert kwargs["batch"] is batch


def test_average_eval_passes_batch():
    camera = MagicMock(height=torch.tensor(2), width=torch.tensor(3))
    batch = {"medium_mask": torch.ones(2, 3, 1, dtype=torch.bool)}
    pipe = _pipeline_with(camera, batch)
    pipe.get_average_image_metrics([(camera, batch)], image_prefix="eval")
    _, kwargs = pipe._model.get_outputs_for_camera.call_args
    assert kwargs["batch"] is batch
