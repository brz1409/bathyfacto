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

"""Camera-optimizer index mapping and medium-mask semantics of the point-cloud exporter.

The export dataset is the ``"data"`` split (every image), while the camera optimizer is indexed
by TRAIN position. Each export image is therefore mapped to its train index by filename, and
held-out images get no correction at all.

A missing medium mask means "no mask, do not restrict", matching the model. An all-False tensor
would mean "no ray is water" and would export an empty point cloud for a run without masks.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from nerfstudio.cameras.rays import RayBundle

from bathyfacto.bathyfacto_datamanager import read_medium_mask
from bathyfacto.export import build_export_to_train_index_map


def _pipeline(train_names, *, raise_on_train=False):
    """Minimal stand-in exposing only what build_export_to_train_index_map touches."""

    def get_dataparser_outputs(split):
        if split == "train":
            if raise_on_train:
                raise RuntimeError("no train split available")
            return SimpleNamespace(image_filenames=[Path("images") / n for n in train_names])
        raise AssertionError(f"unexpected split {split!r}")

    return SimpleNamespace(
        datamanager=SimpleNamespace(dataparser=SimpleNamespace(get_dataparser_outputs=get_dataparser_outputs))
    )


def _outputs(names):
    return SimpleNamespace(image_filenames=[Path("images") / n for n in names])


def test_held_out_images_shift_every_later_index():
    """The core defect: a gap in the middle silently offsets everything after it."""
    data_names = [f"{i:04d}.png" for i in range(1, 6)]  # 0001..0005
    train_names = ["0001.png", "0002.png", "0004.png", "0005.png"]  # 0003 held out

    mapping = build_export_to_train_index_map(_pipeline(train_names), _outputs(data_names))

    assert mapping == {0: 0, 1: 1, 3: 2, 4: 3}
    assert 2 not in mapping, "the held-out image must have no correction at all"
    # Export index 3 is train index 2. Using the export index would have read train 3.
    assert mapping[3] != 3


def test_all_images_in_train_is_the_identity():
    names = ["a.png", "b.png", "c.png"]

    mapping = build_export_to_train_index_map(_pipeline(names), _outputs(names))

    assert mapping == {0: 0, 1: 1, 2: 2}


def test_matching_is_by_filename_not_by_position():
    """Different orderings must still pair correctly."""
    data_names = ["a.png", "b.png", "c.png"]
    train_names = ["c.png", "a.png"]

    mapping = build_export_to_train_index_map(_pipeline(train_names), _outputs(data_names))

    assert mapping == {0: 1, 2: 0}


def test_unavailable_train_split_degrades_to_no_corrections():
    names = ["a.png", "b.png"]

    mapping = build_export_to_train_index_map(_pipeline(names, raise_on_train=True), _outputs(names))

    assert mapping == {}, "without a train split we must skip corrections, not guess indices"


def _ray_bundle(num_rays: int, metadata=None) -> RayBundle:
    return RayBundle(
        origins=torch.zeros(num_rays, 3),
        directions=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_rays, 1),
        pixel_area=torch.ones(num_rays, 1),
        nears=torch.zeros(num_rays, 1),
        fars=torch.ones(num_rays, 1),
        metadata=metadata,
    )


def test_absent_mask_means_no_restriction_not_no_water():
    """The inversion that produced empty point clouds."""
    got = read_medium_mask(_ray_bundle(4), threshold=0.5)

    assert got is None, "a missing medium mask must mean 'do not restrict', matching the model"


def test_present_mask_is_thresholded():
    bundle = _ray_bundle(4, metadata={"medium_mask": torch.tensor([[0.0], [0.4], [0.6], [1.0]])})

    got = read_medium_mask(bundle, threshold=0.5)

    assert got is not None
    assert got.tolist() == [False, False, True, True]
