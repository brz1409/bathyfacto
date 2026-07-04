# BathyFacto Dataset

`simulation-ship` is a BathyFacto-ready repackaging of the DOI dataset
`10.48323/G3CAA-ER166`. It is not a new primary dataset.

- Source title: Synthetic Photogrammetric Dataset for Two-Media 3D Reconstruction: Shipwreck & Terrain
- DOI: `10.48323/G3CAA-ER166`
- Source record: https://researchdata.uibk.ac.at/records/g3caa-er166
- Authors: Frederik Schulte, Markus Brezovsky, Anatol Gunthner, Boris Jutzi, Gottfried Mandlburger, Lukas Winiwarter

Download and install the release package:

```bash
python scripts/download_dataset.py
```

The installed layout is:

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

The release asset is `bathyfacto-simulation-ship-v1.tar.zst`, described by
`datasets/simulation-ship.json`. The repackaging only normalizes the reference
mesh filename to `reference_mesh.ply`; no semantic changes are applied.

When using this data, cite the DOI dataset and the BathyFacto paper.

Before publishing or mirroring the GitHub Release asset, confirm that the UIBK
license or dataset authors permit redistribution of this repackaging.
