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

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "download_dataset.py"


def load_download_module():
    spec = importlib.util.spec_from_file_location("download_dataset", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tar_zst(path: Path, *, include_reference_mesh: bool = True) -> None:
    source_root = path.parent / "source" / "simulation-ship"
    (source_root / "images").mkdir(parents=True)
    (source_root / "medium_masks").mkdir()

    for filename in ("data.npz", "train.npz", "val.npz", "DATASET_PROVENANCE.json"):
        (source_root / filename).write_text(filename, encoding="utf-8")
    if include_reference_mesh:
        (source_root / "reference_mesh.ply").write_text("ply\n", encoding="utf-8")

    for index in range(2):
        (source_root / "images" / f"{index:03d}.png").write_bytes(b"image")
        (source_root / "medium_masks" / f"{index:03d}.png").write_bytes(b"mask")

    compressor = zstandard.ZstdCompressor(level=1)
    with path.open("wb") as compressed_file:
        with compressor.stream_writer(compressed_file) as writer:
            with tarfile.open(fileobj=writer, mode="w|") as archive:
                archive.add(source_root, arcname="simulation-ship")


def write_tar_zst_members(path: Path, members: dict[str, bytes], *, member_type: bytes | None = None) -> None:
    compressor = zstandard.ZstdCompressor(level=1)
    with path.open("wb") as compressed_file:
        with compressor.stream_writer(compressed_file) as writer:
            with tarfile.open(fileobj=writer, mode="w|") as archive:
                for name, content in members.items():
                    info = tarfile.TarInfo(name)
                    if member_type is not None:
                        info.type = member_type
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))


def write_manifest(
    tmp_path: Path,
    archive: Path,
    *,
    archive_root: str = "simulation-ship/",
    asset_name: str | None = None,
    sha256: str | None = None,
) -> Path:
    manifest = {
        "asset_name": asset_name if asset_name is not None else archive.name,
        "url": archive.as_uri(),
        "sha256": sha256 if sha256 is not None else sha256_file(archive),
        "archive_root": archive_root,
        "target_dir": "data/simulation-ship",
        "expected_files": [
            "data.npz",
            "train.npz",
            "val.npz",
            "reference_mesh.ply",
            "DATASET_PROVENANCE.json",
        ],
        "expected_counts": {
            "images/*.png": 2,
            "medium_masks/*.png": 2,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_downloads_valid_archive_to_output_dir_and_cache(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "bathyfacto-simulation-ship-v1.tar.zst"
    write_tar_zst(archive)
    manifest = write_manifest(tmp_path, archive)
    output_dir = tmp_path / "installed"
    cache_dir = tmp_path / "downloads"

    download_dataset.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
        ]
    )

    assert (cache_dir / archive.name).is_file()
    assert (output_dir / "data.npz").is_file()
    assert (output_dir / "reference_mesh.ply").is_file()
    assert len(list((output_dir / "images").glob("*.png"))) == 2
    assert len(list((output_dir / "medium_masks").glob("*.png"))) == 2


def test_existing_target_aborts_without_force(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "bathyfacto-simulation-ship-v1.tar.zst"
    write_tar_zst(archive)
    manifest = write_manifest(tmp_path, archive)
    output_dir = tmp_path / "installed"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(SystemExit):
        download_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--cache-dir",
                str(tmp_path / "downloads"),
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_existing_target_is_replaced_with_force(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "bathyfacto-simulation-ship-v1.tar.zst"
    write_tar_zst(archive)
    manifest = write_manifest(tmp_path, archive)
    output_dir = tmp_path / "installed"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    download_dataset.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(tmp_path / "downloads"),
            "--force",
        ]
    )

    assert not (output_dir / "old.txt").exists()
    assert (output_dir / "reference_mesh.ply").is_file()


def test_bad_cache_hash_aborts_without_force_download(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "bathyfacto-simulation-ship-v1.tar.zst"
    write_tar_zst(archive)
    manifest = write_manifest(tmp_path, archive)
    cache_dir = tmp_path / "downloads"
    cache_dir.mkdir()
    (cache_dir / archive.name).write_bytes(b"not the archive")

    with pytest.raises(SystemExit):
        download_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "installed"),
                "--cache-dir",
                str(cache_dir),
            ]
        )


def test_force_download_replaces_bad_cache_and_installs_dataset(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "bathyfacto-simulation-ship-v1.tar.zst"
    write_tar_zst(archive)
    manifest = write_manifest(tmp_path, archive)
    cache_dir = tmp_path / "downloads"
    cache_dir.mkdir()
    cache_path = cache_dir / archive.name
    cache_path.write_bytes(b"not the archive")

    download_dataset.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "installed"),
            "--cache-dir",
            str(cache_dir),
            "--force-download",
        ]
    )

    assert sha256_file(cache_path) == sha256_file(archive)
    assert (tmp_path / "installed" / "reference_mesh.ply").is_file()


def test_pathful_asset_name_is_rejected_before_cache_path_is_used(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "bathyfacto-simulation-ship-v1.tar.zst"
    write_tar_zst(archive)
    manifest = write_manifest(tmp_path, archive, asset_name="../escape.tar.zst")

    with pytest.raises(SystemExit):
        download_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "installed"),
                "--cache-dir",
                str(tmp_path / "downloads"),
            ]
        )

    assert not (tmp_path / "escape.tar.zst").exists()


def test_missing_required_file_is_rejected_by_validation(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "bathyfacto-simulation-ship-v1.tar.zst"
    write_tar_zst(archive, include_reference_mesh=False)
    manifest = write_manifest(tmp_path, archive)

    with pytest.raises(SystemExit):
        download_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "installed"),
                "--cache-dir",
                str(tmp_path / "downloads"),
            ]
        )


@pytest.mark.parametrize(
    "member_name",
    [
        "/simulation-ship/data.npz",
        "simulation-ship/../escape.txt",
        "other-root/data.npz",
    ],
)
def test_extract_rejects_unsafe_archive_members(tmp_path: Path, member_name: str) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "unsafe.tar.zst"
    write_tar_zst_members(archive, {member_name: b"content"})

    with pytest.raises(SystemExit):
        download_dataset.extract_tar_zst(archive, tmp_path / "extract", "simulation-ship/")


def test_extract_rejects_unsafe_archive_root_without_deleting_output_parent(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    parent = tmp_path / "parent"
    parent.mkdir()
    sentinel = parent / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    archive = tmp_path / "unsafe-root.tar.zst"
    write_tar_zst_members(
        archive,
        {
            "./data.npz": b"data",
            "./train.npz": b"train",
            "./val.npz": b"val",
            "./DATASET_PROVENANCE.json": b"{}",
            "./images/000.png": b"image",
            "./images/001.png": b"image",
            "./medium_masks/000.png": b"mask",
            "./medium_masks/001.png": b"mask",
        },
    )
    manifest = write_manifest(tmp_path, archive, archive_root=".")

    with pytest.raises(SystemExit):
        download_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(parent / "installed"),
                "--cache-dir",
                str(tmp_path / "downloads"),
                "--force",
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_extract_rejects_parent_relative_archive_root(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "unsafe-root.tar.zst"
    write_tar_zst_members(archive, {"../x/data.npz": b"data"})

    with pytest.raises(SystemExit):
        download_dataset.extract_tar_zst(archive, tmp_path / "extract", "../x")


def test_dangerous_output_dir_force_is_rejected_before_deletion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "bathyfacto-simulation-ship-v1.tar.zst"
    write_tar_zst(archive)
    manifest = write_manifest(tmp_path, archive)
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        download_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                ".",
                "--cache-dir",
                str(tmp_path / "downloads"),
                "--force",
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_extract_rejects_tar_special_members(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "special.tar.zst"
    write_tar_zst_members(archive, {"simulation-ship/fifo": b""}, member_type=tarfile.FIFOTYPE)

    with pytest.raises(SystemExit):
        download_dataset.extract_tar_zst(archive, tmp_path / "extract", "simulation-ship/")


def test_force_does_not_replace_existing_target_when_validation_fails(tmp_path: Path) -> None:
    download_dataset = load_download_module()
    archive = tmp_path / "bathyfacto-simulation-ship-v1.tar.zst"
    write_tar_zst(archive, include_reference_mesh=False)
    manifest = write_manifest(tmp_path, archive)
    output_dir = tmp_path / "installed"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(SystemExit):
        download_dataset.main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--cache-dir",
                str(tmp_path / "downloads"),
                "--force",
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
