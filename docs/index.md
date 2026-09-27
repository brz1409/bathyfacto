# BathyFacto documentation

BathyFacto is a refraction-aware two-media NeRF implementation for
bathymetric reconstruction, described in the accompanying publication. This
repository is a Nerfstudio plugin and focuses on the model, the data format,
the training command, and the point-cloud export needed to reproduce the
paper results.

## Contents

- [Installation](installation.md)
- [Dataset format](data.md)
- [Training](training.md)
- [Point-cloud export](export.md)
- [Reproducibility](reproducibility.md)

The dataset is not stored in the repository. Run
`python scripts/download_dataset.py` to fetch the simulation-ship dataset
(see [DATASET.md](../DATASET.md)), or prepare your own NPZ dataset with camera,
image, mask, water-surface, and normalization metadata using
`bathyfacto-build-dataset`.
