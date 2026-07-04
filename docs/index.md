# BathyFacto Documentation

BathyFacto is the paper-only two-media NeRF implementation used for
bathymetric reconstruction in the accompanying publication. This repository is
based on Nerfstudio and keeps the public code path focused on the model, data
format, training command, and point-cloud export needed to reproduce the paper
results.

## Contents

- [Installation](installation.md)
- [Dataset Format](data.md)
- [Training](training.md)
- [Point-Cloud Export](export.md)
- [Reproducibility](reproducibility.md)

The dataset is not included in this repository. The public release expects a
prepared NPZ dataset with camera, image, mask, water-surface, and normalization
metadata.
