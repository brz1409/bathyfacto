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

"""Tests for scripts/download_dataset.py against a small local archive."""

from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path

import pytest
import zstandard

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_dataset.py"


def load_script():
    spec = importlib.util.spec_from_file_location("download_dataset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(tmp_path, monkeypatch):
    """The script with its archive already in the cache and the checksum pointed at it."""
    source = tmp_path / "source" / "simulation-ship"
    (source / "images").mkdir(parents=True)
    (source / "data.npz").write_bytes(b"npz")
    (source / "images" / "0001.png").write_bytes(b"png")

    module = load_script()
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / module.URL.rsplit("/", 1)[-1]
    with archive.open("wb") as file, zstandard.ZstdCompressor().stream_writer(file) as writer:
        with tarfile.open(fileobj=writer, mode="w|") as tar:
            tar.add(source, arcname="simulation-ship")
    monkeypatch.setattr(module, "SHA256", module.sha256_file(archive))
    return module


def test_installs_dataset(script, tmp_path):
    out = tmp_path / "data" / "simulation-ship"
    script.main(["--cache-dir", str(tmp_path / "cache"), "--output-dir", str(out)])
    assert (out / "data.npz").read_bytes() == b"npz"
    assert (out / "images" / "0001.png").is_file()


def test_refuses_to_overwrite_without_force(script, tmp_path):
    out = tmp_path / "data" / "simulation-ship"
    out.mkdir(parents=True)
    with pytest.raises(SystemExit, match="already exists"):
        script.main(["--cache-dir", str(tmp_path / "cache"), "--output-dir", str(out)])


def test_force_replaces_existing_dataset(script, tmp_path):
    out = tmp_path / "data" / "simulation-ship"
    out.mkdir(parents=True)
    (out / "stale.txt").write_text("old")
    script.main(["--cache-dir", str(tmp_path / "cache"), "--output-dir", str(out), "--force"])
    assert not (out / "stale.txt").exists()
    assert (out / "data.npz").is_file()


def test_bad_checksum_deletes_archive(script, tmp_path, monkeypatch):
    monkeypatch.setattr(script, "SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="Checksum mismatch"):
        script.main(["--cache-dir", str(tmp_path / "cache"), "--output-dir", str(tmp_path / "out")])
    assert not any((tmp_path / "cache").iterdir())
    assert not (tmp_path / "out").exists()
