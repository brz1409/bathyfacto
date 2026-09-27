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

"""Plugin-style BathyNerf dataparser registration, kept out of ``bathyfacto_config``.

Nerfstudio's dataparser discovery runs from inside ``nerfstudio.pipelines.base_pipeline``'s own
partial import, so the ``bathynerf`` entry point must resolve to a module that needs no
``VanillaPipelineConfig``/``TrainerConfig`` (unlike ``bathyfacto_config``, which imports this
module to re-export ``BathyNerfDataParser``), or that import raises ``ImportError: cannot
import name 'VanillaPipelineConfig' from partially initialized module``.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from bathyfacto.bathyfacto_dataparser import BathyNerfDataParserConfig

BathyNerfDataParser = DataParserSpecification(
    config=BathyNerfDataParserConfig(),
    description="BathyFacto NPZ dataset (water plane, medium masks, normalization metadata).",
)
