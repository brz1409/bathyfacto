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

"""Tests for Bathy plugin-style method and dataparser registrations.

Covers exactly the 3 shipped method configs:
  - bathyfacto   (paper model; supports --disable-refraction ablation)
  - nerfacto-bathy-compare
  - nerfacto-bathy-compare-camopt
"""

import os

from nerfstudio.configs.bathy_method_configs import (
    BathyFactoMethod,
    NerfactoBathyCompareCamoptMethod,
    NerfactoBathyCompareMethod,
)
from nerfstudio.data.dataparsers.bathynerf_dataparser import BathyNerfDataParserConfig
from nerfstudio.plugins import registry
from nerfstudio.plugins.registry_dataparser import discover_dataparsers


def test_discover_bathy_methods_from_environment_variable():
    """The 3 kept Bathy methods are loadable via the documented plugin import path."""
    old_env = os.environ.get("NERFSTUDIO_METHOD_CONFIGS")
    try:
        os.environ["NERFSTUDIO_METHOD_CONFIGS"] = ",".join(
            [
                "bathyfacto-env=nerfstudio.configs.bathy_method_configs:BathyFactoMethod",
                "nerfacto-bathy-compare-env=nerfstudio.configs.bathy_method_configs:NerfactoBathyCompareMethod",
                "nerfacto-bathy-compare-camopt-env=nerfstudio.configs.bathy_method_configs:NerfactoBathyCompareCamoptMethod",
            ]
        )
        methods, _ = registry.discover_methods()
        assert "bathyfacto-env" in methods
        assert methods["bathyfacto-env"].method_name == "bathyfacto"
        assert "nerfacto-bathy-compare-env" in methods
        assert methods["nerfacto-bathy-compare-env"].method_name == "nerfacto-bathy-compare"
        assert "nerfacto-bathy-compare-camopt-env" in methods
        assert methods["nerfacto-bathy-compare-camopt-env"].method_name == "nerfacto-bathy-compare-camopt"
    finally:
        if old_env is not None:
            os.environ["NERFSTUDIO_METHOD_CONFIGS"] = old_env
        else:
            del os.environ["NERFSTUDIO_METHOD_CONFIGS"]


def test_bathyfacto_method_spec():
    """BathyFactoMethod has the expected method name."""
    assert BathyFactoMethod.config.method_name == "bathyfacto"


def test_nerfacto_compare_method_specs():
    """Nerfacto comparison methods have the expected method names."""
    assert NerfactoBathyCompareMethod.config.method_name == "nerfacto-bathy-compare"
    assert NerfactoBathyCompareCamoptMethod.config.method_name == "nerfacto-bathy-compare-camopt"


def test_bathyfacto_disable_refraction_override():
    """BathyFacto config has a disable_refraction field (supports Table 2 ablation)."""
    from nerfstudio.models.bathyfacto import BathyFactoModelConfig

    cfg = BathyFactoModelConfig()
    assert hasattr(cfg, "disable_refraction"), "BathyFactoModelConfig must have disable_refraction field"
    assert cfg.disable_refraction is False, "disable_refraction must default to False"


def test_discover_bathy_dataparser_from_environment_variable():
    """The BathyNerf dataparser spec resolves through the documented plugin import path."""
    old_env = os.environ.get("NERFSTUDIO_DATAPARSER_CONFIGS")
    try:
        os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = (
            "bathynerf-env=nerfstudio.configs.bathy_dataparser_plugin:BathyNerfDataParser"
        )
        dataparsers, _ = discover_dataparsers()
        assert "bathynerf-env" in dataparsers
        assert isinstance(dataparsers["bathynerf-env"], BathyNerfDataParserConfig)
    finally:
        if old_env is not None:
            os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = old_env
        else:
            del os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"]
