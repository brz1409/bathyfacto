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

"""Dataset builder console messages: one ``bathyfacto  dataset:`` line per event.

Covers ``bathyfacto.build_dataset`` and its two helper modules (``bathyfacto.metashape_xml``,
``bathyfacto.normalization``). Messages are captured by swapping in a recorder for each module's
``CONSOLE``, so rich's own formatting cannot corrupt the assertions.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import bathyfacto.build_dataset as build_dataset_module
import bathyfacto.metashape_xml as metashape_xml_module
import bathyfacto.normalization as normalization_module
from bathyfacto.build_dataset import cleanup_existing_outputs
from bathyfacto.metashape_xml import match_mask_filenames_to_images
from bathyfacto.normalization import select_normalization_rotation


@pytest.fixture
def metashape_recorder(record_console):
    return record_console(metashape_xml_module)


@pytest.fixture
def normalization_recorder(record_console):
    return record_console(normalization_module)


@pytest.fixture
def builder_recorder(record_console):
    return record_console(build_dataset_module)


def test_match_mask_filenames_reports_one_summary_line(tmp_path, metashape_recorder):
    """`metashape_xml.match_mask_filenames_to_images` reports mask count and match count in one line."""
    mask_dir = tmp_path / "medium_masks"
    mask_dir.mkdir()
    for stem in ("0001", "0002"):
        Image.fromarray(np.full((4, 4), 255, dtype=np.uint8), mode="L").save(mask_dir / f"{stem}.png")

    match_mask_filenames_to_images(["images/0001.png", "images/0002.png"], mask_dir)

    assert metashape_recorder.messages == [
        f"bathyfacto  dataset: 2 mask files found in {mask_dir}, 2/2 images matched by filename stem"
    ]


def test_select_normalization_rotation_reports_which_branch_was_taken(normalization_recorder):
    """Both branches (georeferenced and the Rodrigues fallback) name the branch they took."""
    identity = np.eye(3, dtype=np.float32)
    non_identity = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    water_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    select_normalization_rotation(non_identity, water_normal)
    select_normalization_rotation(identity, water_normal)

    assert normalization_recorder.messages == [
        "bathyfacto  dataset: scene normalization rotation set to chunk_rotation (georeferenced)",
        "bathyfacto  dataset: scene normalization rotation set to align_water_to_horizontal "
        "(non-georeferenced fallback)",
    ]


def test_cleanup_existing_outputs_reports_one_summary_line(tmp_path, builder_recorder):
    """All removals of one cleanup pass are reported in a single line."""
    (tmp_path / "train.npz").write_bytes(b"")
    (tmp_path / "val.npz").write_bytes(b"")
    args = SimpleNamespace(num_downscales=0, medium_mask_dir=None)

    cleanup_existing_outputs(args=args, output_dir=tmp_path)

    assert builder_recorder.messages == [
        f"bathyfacto  dataset: removed 2 existing outputs in {tmp_path}: train.npz, val.npz"
    ]


def test_cleanup_existing_outputs_reports_when_nothing_to_remove(tmp_path, builder_recorder):
    args = SimpleNamespace(num_downscales=0, medium_mask_dir=None)

    cleanup_existing_outputs(args=args, output_dir=tmp_path)

    assert builder_recorder.messages == [f"bathyfacto  dataset: no existing outputs to remove in {tmp_path}"]
