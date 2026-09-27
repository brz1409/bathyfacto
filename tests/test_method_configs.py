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

"""The registered methods are the paper configurations (Table 2)."""

from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins import registry
from nerfstudio.plugins.registry_dataparser import discover_dataparsers

from bathyfacto.bathyfacto_config import BathyFactoMethod, BathyNerfDataParser, NerfactoBathyCompareMethod
from bathyfacto.bathyfacto_datamanager import BathyDataManagerConfig
from bathyfacto.bathyfacto_dataparser import BathyNerfDataParserConfig
from bathyfacto.bathyfacto_model import BathyFactoModelConfig
from bathyfacto.bathyfacto_pipeline import BathyPipelineConfig


def test_entry_points_register_exactly_the_paper_methods():
    methods, _ = registry.discover_methods()
    assert {"bathyfacto", "nerfacto-bathy-compare"} <= set(methods)
    # No other variant of the Nerfacto comparison baseline is registered (e.g. a
    # camera-optimizer-on variant): the public release ships exactly one.
    compare_variants = {name for name in methods if name.startswith("nerfacto-bathy-compare")}
    assert compare_variants == {"nerfacto-bathy-compare"}


def test_dataparser_is_registered():
    dataparsers, _ = discover_dataparsers()
    assert isinstance(dataparsers["bathynerf"], BathyNerfDataParserConfig)
    assert isinstance(BathyNerfDataParser.config, BathyNerfDataParserConfig)


def test_bathyfacto_defaults_match_the_paper():
    cfg = BathyFactoMethod.config
    assert cfg.method_name == "bathyfacto"
    assert cfg.max_num_iterations == 100000
    assert isinstance(cfg.pipeline, BathyPipelineConfig)
    assert isinstance(cfg.pipeline.datamanager, BathyDataManagerConfig)
    assert cfg.pipeline.datamanager.train_num_rays_per_batch == 4096
    model = cfg.pipeline.model
    assert isinstance(model, BathyFactoModelConfig)
    assert model.camera_optimizer.mode == "off"
    assert model.disable_refraction is False
    assert "camera_opt" in cfg.optimizers
    for group in ("proposal_networks", "fields"):
        sched = cfg.optimizers[group]["scheduler"]
        assert (sched.lr_final, sched.max_steps) == (0.0001, 200000)


def test_nerfacto_baseline():
    cfg = NerfactoBathyCompareMethod.config
    assert cfg.method_name == "nerfacto-bathy-compare"
    assert type(cfg.pipeline) is VanillaPipelineConfig
    assert isinstance(cfg.pipeline.datamanager, BathyDataManagerConfig)
    assert isinstance(cfg.pipeline.model, NerfactoModelConfig)
    assert cfg.pipeline.model.camera_optimizer.mode == "off"
