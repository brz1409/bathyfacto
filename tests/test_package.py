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

"""The package runs against the pinned upstream Nerfstudio commit."""

import importlib.metadata
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NERFSTUDIO_COMMIT = "50e0e3c70c775e89333256213363badbf074f29d"


def test_nerfstudio_is_the_pinned_install():
    """Nerfstudio must be the install of the pinned commit, not PyPI 1.1.5 or a source checkout.

    PyPI 1.1.5 reports the same version number as the pinned commit, so the commit is read from
    the install record pip writes for a VCS requirement.
    """
    import nerfstudio

    direct_url = importlib.metadata.distribution("nerfstudio").read_text("direct_url.json")
    assert direct_url is not None, "nerfstudio was not installed from the pinned git URL"
    assert json.loads(direct_url).get("vcs_info", {}).get("commit_id") == NERFSTUDIO_COMMIT
    path = Path(nerfstudio.__file__).resolve().as_posix()
    assert "site-packages" in path or "src/nerfstudio" in path, path


def test_requires_python_is_the_tested_version():
    """The declared minimum Python is the version the README says the release is tested with."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    readme = (REPO_ROOT / "README.md").read_text()
    declared = re.search(r'^requires-python\s*=\s*">=(\d+\.\d+)"', pyproject, re.MULTILINE)
    tested = re.search(r"Tested with Python (\d+\.\d+)", readme)
    assert declared is not None and tested is not None
    assert declared.group(1) == tested.group(1)
