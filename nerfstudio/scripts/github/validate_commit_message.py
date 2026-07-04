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

"""Validate commit messages used in this repository."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ALLOWED_TYPES = ("feat", "fix", "refactor", "test", "docs", "ci", "chore")
MAX_SUMMARY_LENGTH = 72
MERGE_PREFIXES = ("Merge ", 'Revert "')
AUTOSQUASH_PREFIXES = ("fixup! ", "squash! ")
SUMMARY_RE = re.compile(
    r"^(?P<type>" + "|".join(ALLOWED_TYPES) + r")(\([a-z0-9][a-z0-9._/-]*\))?(!)?: (?P<description>\S.*)$"
)


def extract_summary(text: str) -> str:
    """Return the first non-empty, non-comment line from a commit message."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def validate_summary(summary: str) -> list[str]:
    """Validate the commit summary line."""
    if not summary:
        return ["Commit message is empty."]

    if summary.startswith(MERGE_PREFIXES):
        return []

    for prefix in AUTOSQUASH_PREFIXES:
        if summary.startswith(prefix):
            summary = summary[len(prefix) :]
            break

    errors = []
    if len(summary) > MAX_SUMMARY_LENGTH:
        errors.append(f"Summary is too long ({len(summary)} > {MAX_SUMMARY_LENGTH} characters).")

    match = SUMMARY_RE.fullmatch(summary)
    if match is None:
        errors.append(
            f"Summary must match '<type>(optional-scope): <description>' with one of: {', '.join(ALLOWED_TYPES)}."
        )
        return errors

    description = match.group("description")
    if description.endswith("."):
        errors.append("Summary description should not end with a period.")

    return errors


def validate_commit_message_file(path: Path) -> list[str]:
    """Validate a Git commit message file."""
    summary = extract_summary(path.read_text(encoding="utf-8"))
    return validate_summary(summary)


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit_msg_file", type=Path, help="Path to the Git commit message file.")
    args = parser.parse_args()

    errors = validate_commit_message_file(args.commit_msg_file)
    if not errors:
        return 0

    print("Invalid commit message.")
    for error in errors:
        print(f"- {error}")
    print("Examples:")
    print("- feat(bathy): add refraction-aware color loss")
    print("- fix(process-data): cast point error to float")
    print("- ci: skip unrelated process-data tests on bathy branches")
    return 1


if __name__ == "__main__":
    sys.exit(main())
