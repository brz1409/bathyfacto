# BathyFacto

BathyFacto is a two-media neural radiance field pipeline for underwater
bathymetric reconstruction from above-water imagery. It is based on Nerfstudio
and keeps this public release focused on the paper model, dataset format,
training commands, and refractive point-cloud export path.

The primary public method is `bathyfacto`. It models Snell refraction at
the air-water interface, uses a single proposal sampler with kinked water-ray
geometry, and conditions appearance on the sampled medium.

## Documentation

- [Overview](docs/index.md)
- [Installation](docs/installation.md)
- [Dataset Format](docs/data.md)
- [Training](docs/training.md)
- [Point-Cloud Export](docs/export.md)
- [Reproducibility](docs/reproducibility.md)

## Quick Start

```bash
pip install -e .
python scripts/download_dataset.py
ns-train bathyfacto --data data/simulation-ship
```

The public dataset workflow is documented in [DATASET.md](DATASET.md). The
release package `bathyfacto-simulation-ship-v1.tar.zst` is repackaged from the
DOI dataset.

## Available Methods

| Method | Purpose |
|-|-|
| `bathyfacto` | Paper model with two-media refraction support |
| `nerfacto-bathy-compare` | Nerfacto comparison method without camera optimization |
| `nerfacto-bathy-compare-camopt` | Nerfacto comparison method with camera optimization |

## Core Modules

| Module | Purpose |
|-|-|
| `nerfstudio/models/bathyfacto.py` | BathyFacto model |
| `nerfstudio/fields/bathy_field.py` | Medium-conditioned radiance field |
| `nerfstudio/data/dataparsers/bathynerf_dataparser.py` | NPZ dataset loader |
| `nerfstudio/model_components/two_media_geometry.py` | Refraction and water-plane geometry |
| `nerfstudio/exporter/bathy_pointcloud_utils.py` | Refractive point-cloud utilities |
| `nerfstudio/configs/bathy_method_configs.py` | Method registration |

## Validation

Run the focused public equivalence test:

```bash
python -m pytest tests/models/test_bathyfacto_equivalence.py -q
```

Run the public test suite:

```bash
python -m pytest tests/ -q
```

The public release candidate should also be validated by a confirming training
run against the paper baseline before being treated as release-ready.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for license and
upstream attribution details.
