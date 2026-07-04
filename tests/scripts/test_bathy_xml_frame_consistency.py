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

"""Frame-consistency diagnostics for the Metashape-XML -> BathyNerf NPZ pipeline.

These tests document the coordinate-chain relationships established during
dataset construction:

  1. The per-camera ``<transform>`` blocks in the Metashape XML are the *exact*
     ground-truth poses mapped through the chunk transform — there is no
     bundle-adjust residual in the camera positions.
  2. The water-surface markers are perfectly planar in the georeferenced/world
     frame.
  3. The Metashape chunk rotation ``R_chunk`` has a measurable magnitude — the
     same order as the rotation the camera optimizer learns during training on
     misaligned poses.
  4. ``bathy_normalization_utils.align_water_to_horizontal`` derives the scene
     normalization rotation from the water normal alone (minimal Rodrigues
     rotation), which discards any "twist about vertical" carried by ``R_chunk``.
     The residual ``R_norm @ R_chunk.T`` is a small gratuitous re-orientation.

The dataset used is the BathyNerf land scene under
``data/Simulation_Land_130_XML_MinimalDataparser/``; tests skip if absent.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
XML_PATH = REPO_ROOT / "data" / "Simulation_Land_130_XML_MinimalDataparser" / "markers.xml"

pytestmark = pytest.mark.skipif(not XML_PATH.exists(), reason=f"Metashape XML fixture missing: {XML_PATH}")


def _rotation_angle_deg(rot: np.ndarray) -> float:
    """Geodesic angle of a 3x3 rotation matrix, in degrees."""
    cos_theta = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _load_xml():
    root = ET.parse(XML_PATH).getroot()
    tn = root.find("./chunk/transform")
    s = float(tn.find("scale").text)
    r_chunk = np.array(list(map(float, tn.find("rotation").text.split()))).reshape(3, 3)
    t_chunk = np.array(list(map(float, tn.find("translation").text.split())))

    cams = sorted(root.findall(".//camera"), key=lambda c: c.get("label", ""))
    cam_transforms = []
    cam_refs = []
    for cam in cams:
        cam_transforms.append(np.array(list(map(float, cam.find("transform").text.split()))).reshape(4, 4))
        ref = cam.find("reference")
        cam_refs.append(np.array([float(ref.get("x")), float(ref.get("y")), float(ref.get("z"))]))

    markers = {}
    for m in root.findall("./chunk/markers/marker"):
        ref = m.find("reference")
        if ref is not None:
            markers[m.get("label")] = np.array([float(ref.get("x")), float(ref.get("y")), float(ref.get("z"))])

    return s, r_chunk, t_chunk, np.array(cam_transforms), np.array(cam_refs), markers


def test_camera_transform_equals_chunkinv_of_reference():
    """Each camera <transform> position == (1/s) R_chunk^T (reference - T_chunk), exactly.

    The XML poses carry no estimation error — they are the analytic chunk-local
    image of the true (georeferenced) camera positions.
    """
    s, r_chunk, t_chunk, cam_transforms, cam_refs, _ = _load_xml()
    assert len(cam_transforms) >= 2
    pred_local = np.array([(1.0 / s) * r_chunk.T @ (p - t_chunk) for p in cam_refs])
    actual_local = cam_transforms[:, :3, 3]
    np.testing.assert_allclose(pred_local, actual_local, atol=1e-9)


def test_markers_are_planar_in_world_frame():
    """The water-surface markers lie on an exact horizontal plane in the world frame."""
    _, _, _, _, _, markers = _load_xml()
    pts = np.array(list(markers.values()))
    assert len(pts) >= 3
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    residuals = np.abs((pts - centroid) @ normal)
    assert residuals.max() < 1e-6
    assert _rotation_angle_deg(np.eye(3)) == 0.0  # sanity
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)


def test_chunk_rotation_magnitude_matches_camopt_scale():
    """R_chunk has a rotation of approximately 0.5–1.1 deg."""
    _, r_chunk, _, _, _, _ = _load_xml()
    angle = _rotation_angle_deg(r_chunk)
    assert 0.5 < angle < 1.1, f"unexpected chunk rotation magnitude: {angle:.4f} deg"


def test_align_water_to_horizontal_loses_twist_vs_rchunk():
    """`R_norm` (from the water normal alone) differs from `R_chunk` by a small twist.

    The marker plane in chunk-local coords has normal == R_chunk^T @ [0,0,1].
    `align_water_to_horizontal` maps that back to vertical via the *minimal*
    rotation, so `R_norm @ R_chunk^T` is a non-identity residual rotation
    — a gratuitous re-orientation of the scene frame.
    """
    from nerfstudio.data.utils.bathy_normalization_utils import align_water_to_horizontal

    s, r_chunk, t_chunk, _, _, markers = _load_xml()
    markers_local = np.array([(1.0 / s) * r_chunk.T @ (p - t_chunk) for p in markers.values()])
    centroid = markers_local.mean(axis=0)
    _, _, vh = np.linalg.svd(markers_local - centroid)
    n_local = vh[-1]
    if n_local[2] < 0:
        n_local = -n_local

    # n_local should be exactly R_chunk^T @ [0,0,1]
    np.testing.assert_allclose(n_local, r_chunk.T @ np.array([0.0, 0.0, 1.0]), atol=1e-6)

    r_norm = np.asarray(align_water_to_horizontal(n_local), dtype=np.float64)
    # Both bring the water normal to vertical.
    np.testing.assert_allclose(r_norm @ n_local, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(r_chunk @ n_local, [0.0, 0.0, 1.0], atol=1e-6)

    residual = r_norm @ r_chunk.T
    residual_angle = _rotation_angle_deg(residual)
    # Non-trivial (R_norm != R_chunk) but small.
    assert residual_angle > 1e-3, "expected a measurable twist difference"
    assert residual_angle < 0.5, f"unexpectedly large residual: {residual_angle:.4f} deg"
