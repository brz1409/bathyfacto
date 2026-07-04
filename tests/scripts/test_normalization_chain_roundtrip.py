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

"""NPZ → ``NormalizationChain`` round-trip and R_norm := R_chunk invariant tests."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from nerfstudio.data.utils.bathy_normalization_utils import NormalizationChain


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rotation_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _make_georeferenced_chain() -> NormalizationChain:
    """Georeferenced case: norm_rot == chunk_rot (the post-2026-05 default)."""
    rot = _rotation_z(0.37) @ _rotation_x(0.013)
    return NormalizationChain(
        norm_scale=12.785,
        norm_center=np.array([0.1, -0.2, 0.3], dtype=np.float32),
        norm_rot=rot.astype(np.float32),
        chunk_scale=4.761,
        chunk_rot=rot.astype(np.float32),
        chunk_trans=np.array([5.0, 6.0, 5.0], dtype=np.float32),
    )


def test_from_npz_metadata_long_keys_matches_explicit_fields() -> None:
    """Building from the dataparser's runtime metadata-style dict gives field-wise equality."""
    chain = _make_georeferenced_chain()
    metadata = {
        "normalization_rotation": np.array(chain.norm_rot, dtype=np.float32),
        "normalization_center": np.array(chain.norm_center, dtype=np.float32),
        "normalization_scale": float(chain.norm_scale),
        "chunk_rotation": np.array(chain.chunk_rot, dtype=np.float32),
        "chunk_translation": np.array(chain.chunk_trans, dtype=np.float32),
    }
    rebuilt = NormalizationChain.from_npz_metadata(metadata, dataparser_scale=chain.chunk_scale)

    assert rebuilt.norm_scale == pytest.approx(chain.norm_scale)
    assert rebuilt.chunk_scale == pytest.approx(chain.chunk_scale)
    np.testing.assert_allclose(rebuilt.norm_center, chain.norm_center, atol=0)
    np.testing.assert_allclose(rebuilt.norm_rot, chain.norm_rot, atol=0)
    np.testing.assert_allclose(rebuilt.chunk_rot, chain.chunk_rot, atol=0)
    np.testing.assert_allclose(rebuilt.chunk_trans, chain.chunk_trans, atol=0)


def test_from_npz_metadata_short_keys_also_supported() -> None:
    """Building from the dataclass's own short field names also succeeds."""
    chain = _make_georeferenced_chain()
    metadata = {
        "norm_rot": np.array(chain.norm_rot, dtype=np.float32),
        "norm_center": np.array(chain.norm_center, dtype=np.float32),
        "norm_scale": float(chain.norm_scale),
        "chunk_rot": np.array(chain.chunk_rot, dtype=np.float32),
        "chunk_trans": np.array(chain.chunk_trans, dtype=np.float32),
        "chunk_scale": float(chain.chunk_scale),
    }
    rebuilt = NormalizationChain.from_npz_metadata(metadata)

    assert rebuilt.norm_scale == pytest.approx(chain.norm_scale)
    np.testing.assert_allclose(rebuilt.norm_rot, chain.norm_rot, atol=0)


def test_from_npz_metadata_missing_keys_raises_keyerror() -> None:
    metadata = {
        "normalization_rotation": np.eye(3, dtype=np.float32),
        # normalization_center missing on purpose
        "normalization_scale": 1.0,
        "chunk_rotation": np.eye(3, dtype=np.float32),
        "chunk_translation": np.zeros(3, dtype=np.float32),
    }
    with pytest.raises(KeyError, match="normalization_center"):
        NormalizationChain.from_npz_metadata(metadata, dataparser_scale=1.0)


def test_post_init_rejects_mismatched_r_norm_and_r_chunk() -> None:
    """For georeferenced data (chunk_rot != I), norm_rot must equal chunk_rot."""
    chunk_rot = _rotation_z(0.5)
    bad_norm_rot = _rotation_z(0.5 + 1e-3)  # differs by > atol=1e-6
    with pytest.raises(ValueError, match="invariant violated"):
        NormalizationChain(
            norm_scale=1.0,
            norm_center=np.zeros(3, dtype=np.float32),
            norm_rot=bad_norm_rot.astype(np.float32),
            chunk_scale=1.0,
            chunk_rot=chunk_rot.astype(np.float32),
            chunk_trans=np.zeros(3, dtype=np.float32),
        )


def test_post_init_allows_identity_chunk_with_arbitrary_norm_rot() -> None:
    """Non-georeferenced fallback: chunk_rot == I, norm_rot from align_water_to_horizontal."""
    # Should NOT raise — this is the documented fallback (see select_normalization_rotation).
    chain = NormalizationChain(
        norm_scale=1.0,
        norm_center=np.zeros(3, dtype=np.float32),
        norm_rot=_rotation_z(0.4).astype(np.float32),
        chunk_scale=1.0,
        chunk_rot=np.eye(3, dtype=np.float32),
        chunk_trans=np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_allclose(chain.chunk_rot, np.eye(3), atol=1e-6)


def test_post_init_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="norm_center must have shape"):
        NormalizationChain(
            norm_scale=1.0,
            norm_center=np.zeros(4, dtype=np.float32),  # wrong shape
            norm_rot=np.eye(3, dtype=np.float32),
            chunk_scale=1.0,
            chunk_rot=np.eye(3, dtype=np.float32),
            chunk_trans=np.zeros(3, dtype=np.float32),
        )


def test_frozen_dataclass_blocks_field_mutation() -> None:
    chain = _make_georeferenced_chain()
    with pytest.raises(FrozenInstanceError):
        chain.norm_scale = 99.0  # type: ignore[misc]
