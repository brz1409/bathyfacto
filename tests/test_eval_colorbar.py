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

"""The colour bar drawn onto the accumulation and depth eval images.

At the published image size (1024x768) the bar is pinned byte for byte to the output of the code
that produced the paper's eval images. Images too small for the bar still come back with their
shape unchanged instead of raising.
"""

import hashlib

import pytest
import torch

from bathyfacto.bathyfacto_model import add_colorbar_to_tensor

_PINNED_1024x768 = {
    ("Depth (Meters)", 1.23, 7.89): "3c7885757832f672ac985fadbfd010115840b8cbda0ee974c5e49f3d501246ba",
    ("Accumulation (Opacity)", 0.0, 1.0): "85e040c48c0a2c5ac5df75bd92de7acd3f68cda1c4acad9e9560de19ce35913a",
}
"""SHA-256 of the uint8 output, recorded from the code before small images were handled."""


def _gradient_image(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(0, 1, height)[:, None, None].expand(height, width, 1)
    x = torch.linspace(0, 1, width)[None, :, None].expand(height, width, 1)
    return torch.cat([y, x, 0.5 * (x + y)], dim=-1)


@pytest.mark.parametrize("title, min_val, max_val", list(_PINNED_1024x768))
def test_colour_bar_at_the_published_size_is_unchanged(title, min_val, max_val):
    out = add_colorbar_to_tensor(_gradient_image(768, 1024), title, min_val, max_val)

    as_uint8 = (out * 255).round().to(torch.uint8).numpy()
    assert hashlib.sha256(as_uint8.tobytes()).hexdigest() == _PINNED_1024x768[(title, min_val, max_val)]


@pytest.mark.parametrize("height, width", [(48, 64), (8, 8), (114, 160), (1, 1)])
def test_small_images_keep_their_shape_and_value_range(height, width):
    image = _gradient_image(height, width)

    out = add_colorbar_to_tensor(image, "Depth (Meters)", 0.0, 1.0)

    assert out.shape == image.shape
    assert torch.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
