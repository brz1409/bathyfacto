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

"""Download the simulation-ship dataset to data/simulation-ship.

The archive is a BathyFacto-ready repackaging of DOI 10.48323/G3CAA-ER166,
see DATASET.md.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import zstandard

URL = "https://github.com/brz1409/bathyfacto/releases/download/dataset-v1/bathyfacto-simulation-ship-v1.tar.zst"
SHA256 = "9ed4625132be3a7a06e657a7eb9f62b355d244b3793a9b44265b773644df0bfb"
ARCHIVE_ROOT = "simulation-ship"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/simulation-ship"))
    parser.add_argument("--cache-dir", type=Path, default=Path("downloads"))
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = parser.parse_args(argv)

    if args.output_dir.exists() and not args.force:
        raise SystemExit(f"{args.output_dir} already exists, use --force to replace it")

    archive = args.cache_dir / URL.rsplit("/", 1)[-1]
    if not archive.exists():
        print(f"Downloading {URL}")
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, archive)
    if sha256_file(archive) != SHA256:
        archive.unlink()
        raise SystemExit(f"Checksum mismatch for {archive}, deleted it. Run the script again to re-download.")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as tmp:
        with archive.open("rb") as file, zstandard.ZstdDecompressor().stream_reader(file) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                # The "data" filter rejects absolute paths and "..", on Pythons that have it.
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(tmp, filter="data")
                else:
                    tar.extractall(tmp)
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        shutil.move(str(Path(tmp) / ARCHIVE_ROOT), str(args.output_dir))

    print(f"Dataset installed at {args.output_dir}")


if __name__ == "__main__":
    main()
