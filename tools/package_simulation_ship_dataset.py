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

"""Build the maintainer release archive for the simulation ship dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import zstandard

PACKAGE_NAME = "simulation-ship"
PACKAGE_VERSION = "v1"
RELEASE_TAG = "dataset-v1"
ASSET_NAME = "bathyfacto-simulation-ship-v1.tar.zst"
GITHUB_REPO = "brz1409/bathyfacto"
SOURCE_TITLE = "Synthetic Photogrammetric Dataset for Two-Media 3D Reconstruction: Shipwreck & Terrain"
SOURCE_DOI = "10.48323/G3CAA-ER166"
SOURCE_RECORD = "https://researchdata.uibk.ac.at/records/g3caa-er166"
SOURCE_AUTHORS = [
    "Frederik Schulte",
    "Markus Brezovsky",
    "Anatol Gunthner",
    "Boris Jutzi",
    "Gottfried Mandlburger",
    "Lukas Winiwarter",
]
TARGET_DIR = "data/simulation-ship"
EXPECTED_IMAGE_COUNT = 130
EXPECTED_MASK_COUNT = 130

ARCHIVE_ROOT = f"{PACKAGE_NAME}/"
DEFAULT_SOURCE_DIR = "../../data/Simulation_Land_130_XML_MinimalDataparser"
DEFAULT_OUTPUT_DIR = "dist"
DEFAULT_MANIFEST = "datasets/simulation-ship.json"
REQUIRED_SOURCE_FILES = ("data.npz", "train.npz", "val.npz", "reference_mesh_ein_boden.ply")
EXPECTED_FILES = ["data.npz", "train.npz", "val.npz", "reference_mesh.ply", "DATASET_PROVENANCE.json"]
RENAMES = {"reference_mesh_ein_boden.ply": "reference_mesh.ply"}
TAR_MTIME = 0
TAR_DIR_MODE = 0o755
TAR_FILE_MODE = 0o644


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    return parser.parse_args(argv)


def _repo_root_for_defaults() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    git_path = repo_root / ".git"
    if git_path.is_file():
        prefix = "gitdir: "
        gitdir_line = git_path.read_text(encoding="utf-8").strip()
        if gitdir_line.startswith(prefix):
            gitdir = Path(gitdir_line[len(prefix) :])
            if not gitdir.is_absolute():
                gitdir = (repo_root / gitdir).resolve()
            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                return gitdir.parent.parent.parent
    return repo_root


def _default_source_dir() -> Path:
    return _repo_root_for_defaults() / DEFAULT_SOURCE_DIR


def _validate_source(source_dir: Path) -> tuple[list[Path], list[Path]]:
    missing_files = [filename for filename in REQUIRED_SOURCE_FILES if not (source_dir / filename).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing required source files in {source_dir}: {', '.join(missing_files)}")

    images = sorted((source_dir / "images").glob("*.png"))
    masks = sorted((source_dir / "medium_masks").glob("*.png"))
    if len(images) != EXPECTED_IMAGE_COUNT:
        raise ValueError(f"Expected {EXPECTED_IMAGE_COUNT} images/*.png files in {source_dir}, found {len(images)}")
    if len(masks) != EXPECTED_MASK_COUNT:
        raise ValueError(f"Expected {EXPECTED_MASK_COUNT} medium_masks/*.png files in {source_dir}, found {len(masks)}")

    return images, masks


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _provenance() -> dict:
    return {
        "source_title": SOURCE_TITLE,
        "source_doi": SOURCE_DOI,
        "source_record": SOURCE_RECORD,
        "source_authors": SOURCE_AUTHORS,
        "renames": RENAMES,
        "semantic_changes": False,
        "generated_by": "tools/package_simulation_ship_dataset.py",
    }


def _stage_dataset(source_dir: Path, stage_root: Path, images: list[Path], masks: list[Path]) -> None:
    archive_root = stage_root / PACKAGE_NAME
    (archive_root / "images").mkdir(parents=True)
    (archive_root / "medium_masks").mkdir()

    shutil.copyfile(source_dir / "data.npz", archive_root / "data.npz")
    shutil.copyfile(source_dir / "train.npz", archive_root / "train.npz")
    shutil.copyfile(source_dir / "val.npz", archive_root / "val.npz")
    shutil.copyfile(source_dir / "reference_mesh_ein_boden.ply", archive_root / "reference_mesh.ply")

    for image in images:
        shutil.copyfile(image, archive_root / "images" / image.name)
    for mask in masks:
        shutil.copyfile(mask, archive_root / "medium_masks" / mask.name)

    _write_json(archive_root / "DATASET_PROVENANCE.json", _provenance())


def _archive_name(path: Path, stage_root: Path) -> str:
    return path.relative_to(stage_root).as_posix()


def _normalized_tarinfo(path: Path, archive_name: str, archive: tarfile.TarFile) -> tarfile.TarInfo:
    tarinfo = archive.gettarinfo(path, arcname=archive_name)
    tarinfo.mtime = TAR_MTIME
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mode = TAR_DIR_MODE if tarinfo.isdir() else TAR_FILE_MODE
    return tarinfo


def _iter_archive_paths(archive_root: Path) -> list[Path]:
    directories = sorted((path for path in archive_root.rglob("*") if path.is_dir()), key=lambda path: path.as_posix())
    files = sorted((path for path in archive_root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    return [archive_root, *directories, *files]


def _write_archive(stage_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=19)
    archive_root = stage_root / PACKAGE_NAME
    with archive_path.open("wb") as compressed_file:
        with compressor.stream_writer(compressed_file) as writer:
            with tarfile.open(fileobj=writer, mode="w|") as archive:
                for path in _iter_archive_paths(archive_root):
                    archive_name = _archive_name(path, stage_root)
                    tarinfo = _normalized_tarinfo(path, archive_name, archive)
                    if path.is_file():
                        with path.open("rb") as file:
                            archive.addfile(tarinfo, fileobj=file)
                    else:
                        archive.addfile(tarinfo)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(sha256: str) -> dict:
    return {
        "name": PACKAGE_NAME,
        "version": PACKAGE_VERSION,
        "package_role": "maintainer-package",
        "source_title": SOURCE_TITLE,
        "source_doi": SOURCE_DOI,
        "source_record": SOURCE_RECORD,
        "source_authors": SOURCE_AUTHORS,
        "release_tag": RELEASE_TAG,
        "asset_name": ASSET_NAME,
        "url": f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{ASSET_NAME}",
        "sha256": sha256,
        "archive_root": ARCHIVE_ROOT,
        "target_dir": TARGET_DIR,
        "expected_files": EXPECTED_FILES,
        "expected_counts": {"images/*.png": EXPECTED_IMAGE_COUNT, "medium_masks/*.png": EXPECTED_MASK_COUNT},
        "renames": RENAMES,
        "semantic_changes": False,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    source_dir = args.source_dir if args.source_dir is not None else _default_source_dir()
    output_dir = args.output_dir
    manifest_path = args.manifest
    archive_path = output_dir / ASSET_NAME

    images, masks = _validate_source(source_dir)
    with TemporaryDirectory() as temporary_directory:
        stage_root = Path(temporary_directory)
        _stage_dataset(source_dir, stage_root, images, masks)
        _write_archive(stage_root, archive_path)

    digest = _sha256(archive_path)
    _write_json(manifest_path, _manifest(digest))

    print(f"Wrote {archive_path}")
    print(f"Wrote {manifest_path}")


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    entrypoint()
