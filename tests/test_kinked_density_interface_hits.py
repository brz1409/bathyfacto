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

"""The proposal sampler's kinked density must respect ``interface_hits``.

``BathyFactoModel.get_outputs`` refracts a sample only when the ray both crosses the water
plane AND is flagged as a water ray (``interface_hits``, which folds in the medium mask). The
kinked density function handed to the proposal sampler applies the same gate. Without it, a
land ray whose geometric path crosses the water plane (routine on any dataset with a shoreline)
would query the proposal network along a refracted path while the field is queried along the
straight one.

``kinked_density_respects_interface_hits`` defaults to the consistent behaviour. Setting it
False applies only the plane-crossing condition, which is what the published runs used.
"""

import torch

from bathyfacto.two_media_geometry import make_kinked_density_fn


def _setup():
    """Two rays pointing straight down at a plane at t=1.0; refraction bends them in +x.

    Ray 0 is a water ray, ray 1 is a land ray that geometrically crosses the plane anyway.
    """
    origins = torch.zeros(2, 3)
    air_dirs = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    interface_param = torch.tensor([1.0, 1.0])
    interface_pts = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    # Deliberately not parallel to air_dirs, so a refracted position is distinguishable.
    water_dirs = torch.tensor([[0.6, 0.0, -0.8], [0.6, 0.0, -0.8]])
    interface_hits = torch.tensor([True, False])

    # One sample before the interface and one past it, per ray.
    positions = torch.tensor(
        [
            [[0.0, 0.0, -0.5], [0.0, 0.0, -2.0]],
            [[0.0, 0.0, -0.5], [0.0, 0.0, -2.0]],
        ]
    )
    return origins, air_dirs, interface_param, interface_pts, water_dirs, interface_hits, positions


def _capture(interface_hits_arg):
    """Run the wrapped density fn and return the positions it actually queried."""
    origins, air_dirs, ip, ipts, wdirs, _hits, positions = _setup()
    seen = {}

    def base_density_fn(pos):
        seen["pos"] = pos.clone()
        return torch.zeros(pos.shape[:-1] + (1,))

    fn = make_kinked_density_fn(base_density_fn, origins, air_dirs, ip, ipts, wdirs, interface_hits_arg)
    fn(positions)
    return seen["pos"], positions


def test_without_the_mask_land_rays_are_refracted_too():
    """Pins the original behaviour, still reachable by setting the flag False."""
    queried, original = _capture(None)

    # Sample 1 is past the interface on both rays -> both get bent into +x.
    assert queried[0, 1, 0] > 0.0, "water ray past the interface must be refracted"
    assert queried[1, 1, 0] > 0.0, "without the mask, the land ray is refracted as well"
    # Samples before the interface are untouched on both rays.
    torch.testing.assert_close(queried[:, 0], original[:, 0])


def test_with_the_mask_only_water_rays_are_refracted():
    _, _, _, _, _, interface_hits, _ = _setup()
    queried, original = _capture(interface_hits)

    assert queried[0, 1, 0] > 0.0, "water ray must still be refracted"
    torch.testing.assert_close(queried[1], original[1], msg="a land ray must be left on its straight path entirely")


def test_the_mask_is_what_changes_the_result():
    """Guards against the argument being accepted but ignored."""
    _, _, _, _, _, interface_hits, _ = _setup()
    without, _ = _capture(None)
    with_mask, _ = _capture(interface_hits)

    assert not torch.allclose(without, with_mask), "passing interface_hits must change the queried positions"
    torch.testing.assert_close(without[0], with_mask[0], msg="water rays must be unaffected by the mask")


def test_model_config_defaults_to_the_consistent_behaviour():
    from bathyfacto.bathyfacto_model import BathyFactoModelConfig

    assert BathyFactoModelConfig().kinked_density_respects_interface_hits is True
