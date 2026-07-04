# Dataset Format

Training data is not shipped with this repository. Install the public
`simulation-ship` package with:

```bash
python scripts/download_dataset.py
```

This installs a BathyFacto-ready repackaging of DOI `10.48323/G3CAA-ER166` at
`data/simulation-ship/`. See [../DATASET.md](../DATASET.md) for provenance,
citation, and redistribution notes.

BathyFacto expects a prepared NPZ dataset directory. The dataparser reads image
arrays, camera calibration and poses, water-surface metadata, medium masks,
validity masks, and normalization metadata that maps between dataset, scene,
and global coordinates.

The installed dataset layout is:

```text
data/simulation-ship/
  data.npz
  train.npz
  val.npz
  reference_mesh.ply
  images/*.png
  medium_masks/*.png
  DATASET_PROVENANCE.json
```

For other compatible datasets, keep the same dataparser contract: NPZ files for
training and evaluation splits, images, masks, water-surface metadata, camera
metadata, and normalization metadata. The retained dataset builder can convert
a suitable Metashape XML export and associated image and mask data into this
format:

```bash
python -m nerfstudio.scripts.create_bathynerf_dataset --help
```

The repository intentionally does not include raw images, generated NPZ files,
meshes, checkpoints, or exported point clouds.
