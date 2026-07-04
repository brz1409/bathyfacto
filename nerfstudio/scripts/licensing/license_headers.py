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

"""Add or validate license headers on Python files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET_ROOT = REPO_ROOT / "nerfstudio"
COPYRIGHT_PATH = Path(__file__).with_name("copyright.txt")
ENCODING_RE = re.compile(r"#.*coding[:=]\s*([-\w.]+)")


def read_text_preserve_newlines(path: Path) -> str:
    """Read a file without normalizing newline characters."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return file.read()


def write_text_preserve_newlines(path: Path, content: str) -> None:
    """Write a file without modifying newline characters."""
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(content)


def header_insert_index(lines: Sequence[str]) -> int:
    """Return the line index where the license header should be inserted."""
    index = 0
    if index < len(lines) and lines[index].startswith("#!"):
        index += 1
    if index < len(lines) and ENCODING_RE.fullmatch(lines[index].rstrip("\r\n")):
        index += 1
    return index


def has_license_header(text: str) -> bool:
    """Return True when a Python file already contains a copyright header."""
    return "Copyright" in text


def detect_newline(text: str) -> str:
    """Reuse the file's newline convention when adding headers."""
    return "\r\n" if "\r\n" in text else "\n"


def load_header(newline: str) -> str:
    """Load the canonical license header using the requested newline style."""
    header = read_text_preserve_newlines(COPYRIGHT_PATH)
    return header.replace("\n", newline)


def add_license_header(text: str, header: str) -> str:
    """Insert the license header after an optional shebang or encoding line."""
    lines = text.splitlines(keepends=True)
    index = header_insert_index(lines)
    return "".join(lines[:index]) + header + "".join(lines[index:])


def resolve_target_paths(raw_paths: Sequence[str]) -> list[Path]:
    """Resolve target Python files from CLI or pre-commit input."""
    if not raw_paths:
        return sorted(DEFAULT_TARGET_ROOT.rglob("*.py"))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        path = Path(raw_path)
        path = path if path.is_absolute() else REPO_ROOT / path
        if not path.exists():
            continue

        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(path.rglob("*.py"))
        elif path.suffix == ".py":
            candidates = [path]
        else:
            candidates = []

        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            resolved.append(candidate)
    return resolved


def process_paths(raw_paths: Sequence[str], *, check: bool) -> int:
    """Validate or update license headers for the requested paths."""
    targets = resolve_target_paths(raw_paths)
    missing_headers = False

    for path in targets:
        text = read_text_preserve_newlines(path)
        if has_license_header(text):
            continue

        missing_headers = True
        try:
            display_path = path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = path
        if check:
            print(f"{display_path} missing copyright header")
            continue

        updated = add_license_header(text, load_header(detect_newline(text)))
        write_text_preserve_newlines(path, updated)
        print(f"Adding license header to {display_path}.")

    if check and missing_headers:
        print("Run 'python nerfstudio/scripts/licensing/license_headers.py' to add missing headers.")
        return 1

    if not missing_headers:
        print("No missing license headers found.")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--check", action="store_true", help="Only report missing headers; do not modify files.")
    parser.add_argument("paths", nargs="*", help="Optional file or directory paths to process.")
    args = parser.parse_args()
    return process_paths(args.paths, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
