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

"""Download and install the public simulation ship dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp

import zstandard

REQUIRED_MANIFEST_KEYS = {
    "asset_name",
    "url",
    "sha256",
    "archive_root",
    "target_dir",
    "expected_files",
    "expected_counts",
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/simulation-ship.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("downloads"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args(argv)


def _required_string(manifest: dict[str, object], key: str) -> str:
    value = manifest[key]
    if not isinstance(value, str):
        raise SystemExit(f"Manifest key '{key}' must be a string")
    return value


def _reject_path_parts(path: PurePosixPath, *, label: str) -> None:
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SystemExit(f"{label} must be a safe relative path")


def _reject_raw_path_parts(path: str, *, label: str) -> None:
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise SystemExit(f"{label} must be a safe relative path")


def load_manifest(path: Path) -> dict[str, object]:
    """Load and validate the dataset manifest."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in manifest {path}: {error}") from error
    except OSError as error:
        raise SystemExit(f"Could not read manifest {path}: {error}") from error

    if not isinstance(data, dict):
        raise SystemExit(f"Manifest {path} must contain a JSON object")

    missing_keys = sorted(REQUIRED_MANIFEST_KEYS - data.keys())
    if missing_keys:
        raise SystemExit(f"Manifest {path} is missing required keys: {', '.join(missing_keys)}")

    return data


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(url: str, destination: Path) -> None:
    """Download an asset to a temporary file and atomically replace the destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response:
            with temporary_destination.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary_destination.replace(destination)
    except Exception:
        temporary_destination.unlink(missing_ok=True)
        raise


def ensure_asset(manifest: dict[str, object], cache_dir: Path, force_download: bool) -> Path:
    """Return a verified local cache path for the manifest asset."""
    asset_name = _safe_asset_name(_required_string(manifest, "asset_name"))
    url = _required_string(manifest, "url")
    expected_hash = _required_string(manifest, "sha256").lower()
    cache_path = cache_dir / asset_name

    if cache_path.exists() and force_download:
        cache_path.unlink()

    if not cache_path.exists():
        download_asset(url, cache_path)

    actual_hash = sha256_file(cache_path)
    if actual_hash == expected_hash:
        return cache_path

    if force_download:
        cache_path.unlink(missing_ok=True)
        download_asset(url, cache_path)
        actual_hash = sha256_file(cache_path)
        if actual_hash == expected_hash:
            return cache_path

    raise SystemExit(
        f"SHA256 mismatch for cached asset {cache_path}: expected {expected_hash}, found {actual_hash}. "
        "Use --force-download to re-download."
    )


def _safe_asset_name(asset_name: str) -> str:
    if (
        not asset_name
        or asset_name in {".", ".."}
        or Path(asset_name).is_absolute()
        or Path(asset_name).name != asset_name
        or "/" in asset_name
        or "\\" in asset_name
    ):
        raise SystemExit(f"Manifest key 'asset_name' must be a plain filename: {asset_name}")
    return asset_name


def _normalized_archive_root(archive_root: str) -> str:
    if archive_root.startswith("/"):
        raise SystemExit("Manifest key 'archive_root' must be a safe relative path")
    normalized = archive_root.rstrip("/")
    if not normalized or normalized in {".", ".."}:
        raise SystemExit("Manifest key 'archive_root' must be a safe relative path")
    _reject_raw_path_parts(normalized, label="Manifest key 'archive_root'")
    _reject_path_parts(PurePosixPath(normalized), label="Manifest key 'archive_root'")
    return normalized


def _is_safe_member(name: str, archive_root: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and (name == archive_root or name.startswith(f"{archive_root}/"))


def _validate_archive_members(archive: Path, archive_root: str) -> None:
    found_root = False
    decompressor = zstandard.ZstdDecompressor()
    with archive.open("rb") as compressed_file:
        with decompressor.stream_reader(compressed_file) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if not _is_safe_member(member.name, archive_root):
                        raise SystemExit(f"Unsafe archive member rejected: {member.name}")
                    if not (member.isfile() or member.isdir()):
                        raise SystemExit(f"Archive member is not a regular file or directory: {member.name}")
                    if member.name == archive_root or member.name.startswith(f"{archive_root}/"):
                        found_root = True
    if not found_root:
        raise SystemExit(f"Archive {archive} does not contain expected root {archive_root}")


def extract_tar_zst(archive: Path, destination_parent: Path, archive_root: str) -> Path:
    """Extract a .tar.zst archive under destination_parent and return its root directory."""
    extracted_root, _ = _extract_tar_zst_to_temporary_dir(archive, destination_parent, archive_root)
    return extracted_root


def _extract_tar_zst_to_temporary_dir(archive: Path, destination_parent: Path, archive_root: str) -> tuple[Path, Path]:
    """Extract a .tar.zst archive and return the extracted root plus its temp dir."""
    normalized_root = _normalized_archive_root(archive_root)
    destination_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(mkdtemp(prefix=".dataset-download-", dir=destination_parent))

    try:
        _validate_archive_members(archive, normalized_root)
        decompressor = zstandard.ZstdDecompressor()
        with archive.open("rb") as compressed_file:
            with decompressor.stream_reader(compressed_file) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(temporary_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    extracted_root = temporary_dir / normalized_root
    if not extracted_root.is_dir():
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise SystemExit(f"Archive {archive} did not extract expected root {normalized_root}")
    return extracted_root, temporary_dir


def validate_dataset(root: Path, manifest: dict[str, object]) -> None:
    """Validate required files and expected glob counts below an extracted dataset root."""
    expected_files = manifest["expected_files"]
    if not isinstance(expected_files, list):
        raise SystemExit("Manifest key 'expected_files' must be a list")
    for expected_file in expected_files:
        if not isinstance(expected_file, str):
            raise SystemExit("Manifest key 'expected_files' must contain strings")
        if not (root / expected_file).is_file():
            raise SystemExit(f"Missing expected dataset file: {expected_file}")

    expected_counts = manifest["expected_counts"]
    if not isinstance(expected_counts, dict):
        raise SystemExit("Manifest key 'expected_counts' must be an object")
    for pattern, expected_count in expected_counts.items():
        if not isinstance(pattern, str):
            raise SystemExit("Manifest key 'expected_counts' must use string patterns")
        if not isinstance(expected_count, int):
            raise SystemExit(f"Expected count for {pattern} must be an integer")
        actual_count = len(list(root.glob(pattern)))
        if actual_count != expected_count:
            raise SystemExit(f"Expected {expected_count} files matching {pattern}, found {actual_count}")


def _target_output_dir(args_output_dir: Path | None, manifest: dict[str, object]) -> Path:
    if args_output_dir is not None:
        return _validate_target_dir(args_output_dir, source="Output directory")
    target_dir = Path(_required_string(manifest, "target_dir"))
    if target_dir.is_absolute():
        raise SystemExit("Manifest key 'target_dir' must be a safe relative path")
    target_dir_posix = target_dir.as_posix()
    _reject_raw_path_parts(target_dir_posix, label="Manifest key 'target_dir'")
    _reject_path_parts(PurePosixPath(target_dir_posix), label="Manifest key 'target_dir'")
    return _validate_target_dir(target_dir, source="Manifest key 'target_dir'")


def _validate_target_dir(target_dir: Path, *, source: str) -> Path:
    expanded = target_dir.expanduser()
    resolved = expanded.resolve(strict=False)
    home = Path.home().resolve()
    if expanded == Path(".") or expanded == Path(""):
        raise SystemExit(f"{source} must not be the current directory")
    if resolved == Path(resolved.anchor):
        raise SystemExit(f"{source} must not be the filesystem root")
    if resolved == Path.cwd().resolve():
        raise SystemExit(f"{source} must not be the current directory")
    if resolved == home:
        raise SystemExit(f"{source} must not be the home directory")
    return target_dir


def _move_validated_dataset(extracted_root: Path, target_dir: Path, force: bool) -> None:
    if target_dir.exists():
        if not force:
            raise SystemExit(f"Target directory already exists: {target_dir}. Use --force to replace it.")
        if not target_dir.is_dir():
            raise SystemExit(f"Target exists but is not a directory: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted_root), str(target_dir))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = load_manifest(args.manifest)
    target_dir = _target_output_dir(args.output_dir, manifest)

    if target_dir.exists() and not args.force:
        raise SystemExit(f"Target directory already exists: {target_dir}. Use --force to replace it.")

    archive = ensure_asset(manifest, args.cache_dir, args.force_download)
    archive_root = _required_string(manifest, "archive_root")
    extracted_root, temporary_dir = _extract_tar_zst_to_temporary_dir(archive, target_dir.parent, archive_root)

    try:
        validate_dataset(extracted_root, manifest)
        _move_validated_dataset(extracted_root, target_dir, args.force)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)

    print(f"Dataset installed at {target_dir}")


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    entrypoint()
