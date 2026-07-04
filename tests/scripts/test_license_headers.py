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

from nerfstudio.scripts.licensing.license_headers import add_license_header, has_license_header, process_paths


def test_add_license_header_preserves_shebang() -> None:
    header = "# Copyright 2026 Nerfstudio\n#\n"
    text = "#!/usr/bin/env python\nprint('hello')\n"

    updated = add_license_header(text, header)

    assert updated == "#!/usr/bin/env python\n# Copyright 2026 Nerfstudio\n#\nprint('hello')\n"
    assert has_license_header(updated)


def test_process_paths_check_mode_reports_missing_header(tmp_path, capsys) -> None:
    target = tmp_path / "missing.py"
    target.write_text("print('hello')\n", encoding="utf-8")

    exit_code = process_paths([str(target)], check=True)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing.py missing copyright header" in captured.out
    assert target.read_text(encoding="utf-8") == "print('hello')\n"


def test_process_paths_adds_header_to_target_file(tmp_path) -> None:
    target = tmp_path / "missing.py"
    target.write_text("print('hello')\n", encoding="utf-8")

    exit_code = process_paths([str(target)], check=False)

    updated = target.read_text(encoding="utf-8")
    assert exit_code == 0
    assert updated.startswith("# Copyright 2022 the Regents of the University of California")
    assert updated.endswith("print('hello')\n")
