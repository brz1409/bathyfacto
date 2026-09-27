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


"""Command line of ``bathyfacto-build-dataset``."""

import sys

import pytest

import bathyfacto.build_dataset as build_dataset_module


def test_data_dir_is_required(monkeypatch, capsys):
    """Without ``--data-dir`` the builder must stop, not read and write inside the installed package."""
    calls = []
    monkeypatch.setattr(build_dataset_module, "main", calls.append)
    monkeypatch.setattr(sys, "argv", ["bathyfacto-build-dataset"])

    with pytest.raises(SystemExit) as excinfo:
        build_dataset_module.entrypoint()

    assert excinfo.value.code == 2
    assert calls == []
    assert "--data-dir" in capsys.readouterr().err
