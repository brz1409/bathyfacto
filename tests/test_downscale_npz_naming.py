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

"""Naming of downscaled NPZ splits: an ``_<factor>x`` suffix in the dataset root.

``bathyfacto-build-dataset`` writes ``train_2x.npz`` next to ``train.npz``, and
``BathyNerfDataParser`` looks for that name first when ``downscale_factor`` is 2.
"""

from pathlib import Path

import pytest

from bathyfacto.bathyfacto_dataparser import BathyNerfDataParser, BathyNerfDataParserConfig


def _parser(data_dir: Path, downscale_factor=None) -> BathyNerfDataParser:
    return BathyNerfDataParser(BathyNerfDataParserConfig(data=data_dir, downscale_factor=downscale_factor))


def _touch(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def test_factor_one_is_unchanged(tmp_path):
    """The un-downscaled path must behave exactly as before."""
    expected = _touch(tmp_path / "train.npz")
    for factor in (None, 1):
        assert _parser(tmp_path, factor)._resolve_split_filepath("train") == (expected, False)


def test_downscaled_split_is_preferred(tmp_path):
    _touch(tmp_path / "train.npz")
    downscaled = _touch(tmp_path / "train_2x.npz")

    assert _parser(tmp_path, 2)._resolve_split_filepath("train") == (downscaled, True)


def test_falls_back_to_unsuffixed_when_downscaled_missing(tmp_path):
    """Datasets built before the naming was unified must still load."""
    plain = _touch(tmp_path / "train.npz")

    assert _parser(tmp_path, 2)._resolve_split_filepath("train") == (plain, False)


def test_test_split_falls_back_to_val_at_the_same_scale(tmp_path):
    _touch(tmp_path / "val.npz")
    val_downscaled = _touch(tmp_path / "val_2x.npz")

    assert _parser(tmp_path, 2)._resolve_split_filepath("test") == (val_downscaled, True)


def test_error_names_every_candidate_tried(tmp_path):
    """A missing split must name both paths it looked for."""
    with pytest.raises(FileNotFoundError) as excinfo:
        _parser(tmp_path, 2)._resolve_split_filepath("train")

    message = str(excinfo.value)
    assert "train_2x.npz" in message
    assert "train.npz" in message
