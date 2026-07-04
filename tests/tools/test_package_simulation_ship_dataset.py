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

import hashlib
import importlib.util
import json
import os
import shutil
import tarfile
from pathlib import Path

import zstandard


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "package_simulation_ship_dataset.py"
    spec = importlib.util.spec_from_file_location("package_simulation_ship_dataset", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source_dataset(source: Path) -> None:
    (source / "images").mkdir(parents=True)
    (source / "medium_masks").mkdir()
    (source / ".scratch").mkdir()
    (source / ".ipynb_checkpoints").mkdir()
    (source / "analysis").mkdir()
    (source / "images" / "outputs").mkdir()

    for filename in ["data.npz", "train.npz", "val.npz", "reference_mesh_ein_boden.ply"]:
        (source / filename).write_bytes(b"dataset")

    png_bytes = b"\x89PNG\r\n\x1a\n"
    for index in range(130):
        (source / "images" / f"{index:03d}.png").write_bytes(png_bytes)
        (source / "medium_masks" / f"{index:03d}.png").write_bytes(png_bytes)

    excluded_files = [
        ".scratch/notes.txt",
        ".ipynb_checkpoints/checkpoint.ipynb",
        "analysis/report.json",
        "images/outputs/render.png",
        "data.bak_2026",
        "scene.metashape_sfm",
        "visualization_scene.png",
        "visualization_scene.html",
        "markers.xml",
        "tie_points.ply",
    ]
    for relative_path in excluded_files:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded", encoding="utf-8")


def _set_tree_mtime(path: Path, mtime: int) -> None:
    for child in path.rglob("*"):
        os.utime(child, (mtime, mtime))
    os.utime(path, (mtime, mtime))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packages_simulation_ship_dataset_with_manifest_and_provenance(tmp_path) -> None:
    module = _load_module()
    source = tmp_path / "source"
    output_dir = tmp_path / "dist"
    manifest_path = tmp_path / "datasets" / "simulation-ship.json"
    _write_source_dataset(source)

    module.main(["--source-dir", str(source), "--output-dir", str(output_dir), "--manifest", str(manifest_path)])

    archive_path = output_dir / "bathyfacto-simulation-ship-v1.tar.zst"
    assert archive_path.exists()

    tar_path = tmp_path / "archive.tar"
    decompressor = zstandard.ZstdDecompressor()
    with archive_path.open("rb") as compressed_file, tar_path.open("wb") as tar_file:
        decompressor.copy_stream(compressed_file, tar_file)

    with tarfile.open(tar_path) as archive:
        names = set(archive.getnames())
        assert "simulation-ship/reference_mesh.ply" in names
        assert "simulation-ship/reference_mesh_ein_boden.ply" not in names
        assert "simulation-ship/DATASET_PROVENANCE.json" in names
        assert "simulation-ship/images/000.png" in names
        assert "simulation-ship/medium_masks/129.png" in names
        assert archive.getmember("simulation-ship").mtime == 0
        assert archive.getmember("simulation-ship").uid == 0
        assert archive.getmember("simulation-ship").gid == 0
        assert archive.getmember("simulation-ship").uname == ""
        assert archive.getmember("simulation-ship").gname == ""
        assert archive.getmember("simulation-ship").mode == 0o755
        assert archive.getmember("simulation-ship/reference_mesh.ply").mtime == 0
        assert archive.getmember("simulation-ship/reference_mesh.ply").uid == 0
        assert archive.getmember("simulation-ship/reference_mesh.ply").gid == 0
        assert archive.getmember("simulation-ship/reference_mesh.ply").uname == ""
        assert archive.getmember("simulation-ship/reference_mesh.ply").gname == ""
        assert archive.getmember("simulation-ship/reference_mesh.ply").mode == 0o644

        excluded_names = {
            "simulation-ship/.scratch/notes.txt",
            "simulation-ship/.ipynb_checkpoints/checkpoint.ipynb",
            "simulation-ship/analysis/report.json",
            "simulation-ship/images/outputs/render.png",
            "simulation-ship/data.bak_2026",
            "simulation-ship/scene.metashape_sfm",
            "simulation-ship/visualization_scene.png",
            "simulation-ship/visualization_scene.html",
            "simulation-ship/markers.xml",
            "simulation-ship/tie_points.ply",
        }
        assert names.isdisjoint(excluded_names)

        provenance_file = archive.extractfile("simulation-ship/DATASET_PROVENANCE.json")
        assert provenance_file is not None
        provenance = json.loads(provenance_file.read().decode("utf-8"))
        assert provenance["semantic_changes"] is False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_doi"] == "10.48323/G3CAA-ER166"
    assert manifest["source_record"] == "https://researchdata.uibk.ac.at/records/g3caa-er166"
    assert manifest["archive_root"] == "simulation-ship/"
    assert manifest["target_dir"] == "data/simulation-ship"
    assert manifest["expected_counts"] == {"images/*.png": 130, "medium_masks/*.png": 130}
    assert (
        manifest["url"] == "https://github.com/brz1409/bathyfacto/releases/download/dataset-v1/"
        "bathyfacto-simulation-ship-v1.tar.zst"
    )
    assert len(manifest["sha256"]) == 64


def test_packages_are_reproducible_when_source_mtimes_change(tmp_path) -> None:
    module = _load_module()
    source = tmp_path / "source"
    _write_source_dataset(source)

    first_output_dir = tmp_path / "first-dist"
    first_manifest_path = tmp_path / "first-datasets" / "simulation-ship.json"
    _set_tree_mtime(source, 1_700_000_000)
    module.main(
        ["--source-dir", str(source), "--output-dir", str(first_output_dir), "--manifest", str(first_manifest_path)]
    )

    second_source = tmp_path / "second-source"
    shutil.copytree(source, second_source)
    _set_tree_mtime(second_source, 1_800_000_000)
    second_output_dir = tmp_path / "second-dist"
    second_manifest_path = tmp_path / "second-datasets" / "simulation-ship.json"
    module.main(
        [
            "--source-dir",
            str(second_source),
            "--output-dir",
            str(second_output_dir),
            "--manifest",
            str(second_manifest_path),
        ]
    )

    first_archive = first_output_dir / "bathyfacto-simulation-ship-v1.tar.zst"
    second_archive = second_output_dir / "bathyfacto-simulation-ship-v1.tar.zst"
    assert _sha256(first_archive) == _sha256(second_archive)
