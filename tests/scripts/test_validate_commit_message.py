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

from nerfstudio.scripts.github.validate_commit_message import validate_summary


def test_validate_summary_accepts_conventional_commit():
    assert validate_summary("fix(process-data): cast point error to float") == []


def test_validate_summary_accepts_merge_commit():
    assert validate_summary("Merge branch 'bathy-dev' into bathy-markus") == []


def test_validate_summary_accepts_autosquash_commit():
    assert validate_summary("fixup! ci: skip unrelated process-data tests on bathy branches") == []


def test_validate_summary_rejects_missing_type():
    assert validate_summary("update bathy workflow")


def test_validate_summary_rejects_trailing_period():
    assert validate_summary("docs: update bathy workflow.")
