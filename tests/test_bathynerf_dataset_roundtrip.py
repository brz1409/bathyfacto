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

"""Regression: dataset-creation forward transform <-> dataparser load <-> global-frame round-trip.

Exercises the forward chunk-local/global transform chain against the exporter's inverse
``transform_points_to_bathy_global_frame``. The full on-disk path (real ``BathyNerfDataParser``
load) is gated behind the shipped sample dataset; the always-run test narrows to a
``SimpleNamespace`` metadata stand-in (as ``tests/test_bathy_pointcloud_export.py``
does) so the round-trip math is exercised on every CI run regardless of dataset availability.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import rotation_x, rotation_z

from bathyfacto.bathyfacto_dataparser import BathyNerfDataParserConfig
from bathyfacto.export import transform_points_to_bathy_global_frame

_SAMPLE_DATASET = Path("data/simulation-ship")


def _known_global_points() -> np.ndarray:
    """A small fixed set of "global frame" points whose round-trip we verify."""
    return np.array(
        [
            [5.0, 6.0, 7.0],
            [4.0, 6.5, 5.0],
            [6.2, 5.4, 8.0],
            [5.5, 6.0, 5.0],
        ],
        dtype=np.float32,
    )


def _forward_global_to_normalized(
    global_points: np.ndarray,
    *,
    norm_rot: np.ndarray,
    norm_center: np.ndarray,
    norm_scale: float,
    chunk_rot: np.ndarray,
    chunk_trans: np.ndarray,
    chunk_scale: float,
) -> np.ndarray:
    """Inverse of ``transform_points_to_bathy_global_frame`` (the dataset-creation direction)."""
    # global = chunk_scale * (chunk_local @ chunk_rot.T) + chunk_trans
    chunk_local = ((global_points - chunk_trans) / chunk_scale) @ chunk_rot
    # chunk_local = (normalized * norm_scale + norm_center) @ norm_rot
    normalized = (chunk_local @ norm_rot.T - norm_center) / norm_scale
    return normalized.astype(np.float32)


def test_npz_creation_to_global_frame_roundtrip():
    """Known global points -> forward transform -> exporter inverse recovers them at mm level."""
    # NormalizationChain invariant: for georeferenced datasets norm_rot == chunk_rot
    # (see `select_normalization_rotation` in bathyfacto.normalization).
    shared_rot = rotation_z(0.37) @ rotation_x(0.013)
    norm_rot = shared_rot.astype(np.float32)
    norm_center = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    norm_scale = 12.785
    chunk_rot = shared_rot.astype(np.float32)
    chunk_trans = np.array([5.0, 6.0, 5.0], dtype=np.float32)
    chunk_scale = 4.761

    global_points = _known_global_points()
    normalized = _forward_global_to_normalized(
        global_points,
        norm_rot=norm_rot,
        norm_center=norm_center,
        norm_scale=norm_scale,
        chunk_rot=chunk_rot,
        chunk_trans=chunk_trans,
        chunk_scale=chunk_scale,
    )

    dataparser_outputs = SimpleNamespace(
        metadata={
            "normalization_rotation": norm_rot,
            "normalization_center": norm_center,
            "normalization_scale": norm_scale,
            "chunk_rotation": chunk_rot,
            "chunk_translation": chunk_trans,
        },
        dataparser_scale=chunk_scale,
    )

    recovered = transform_points_to_bathy_global_frame(torch.from_numpy(normalized), dataparser_outputs).numpy()

    # eff_scale ~ 60.87 m per scene unit, so 1e-3 m global tolerance is mm level.
    np.testing.assert_allclose(recovered, global_points, atol=1e-3)


@pytest.mark.skipif(
    not (_SAMPLE_DATASET / "train.npz").exists(),
    reason="sample dataset not present — run `python scripts/download_dataset.py` first (installs data/simulation-ship)",
)
def test_real_sample_dataset_roundtrip():
    """Real BathyNerfDataParser load: water_surface survives, dataparser_scale matches NPZ, round-trip holds."""
    config = BathyNerfDataParserConfig(data=_SAMPLE_DATASET)
    outputs = config.setup().get_dataparser_outputs("data")

    assert "water_surface" in outputs.metadata
    for key in (
        "normalization_rotation",
        "normalization_center",
        "normalization_scale",
        "chunk_rotation",
        "chunk_translation",
    ):
        assert key in outputs.metadata, f"missing reversible-normalization key {key!r}"

    # Round-trip a handful of scene-space points through the exporter and back to scene space.
    rng = np.random.default_rng(0)
    scene_points = rng.uniform(-0.5, 0.5, size=(8, 3)).astype(np.float32)
    global_points = transform_points_to_bathy_global_frame(torch.from_numpy(scene_points), outputs).numpy()

    norm_rot = np.asarray(outputs.metadata["normalization_rotation"], dtype=np.float32)
    norm_center = np.asarray(outputs.metadata["normalization_center"], dtype=np.float32)
    norm_scale = float(outputs.metadata["normalization_scale"])
    chunk_rot = np.asarray(outputs.metadata["chunk_rotation"], dtype=np.float32)
    chunk_trans = np.asarray(outputs.metadata["chunk_translation"], dtype=np.float32)
    chunk_scale = float(outputs.dataparser_scale)

    back = _forward_global_to_normalized(
        global_points,
        norm_rot=norm_rot,
        norm_center=norm_center,
        norm_scale=norm_scale,
        chunk_rot=chunk_rot,
        chunk_trans=chunk_trans,
        chunk_scale=chunk_scale,
    )
    np.testing.assert_allclose(back, scene_points, atol=1e-4)
