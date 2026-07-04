# Training

The primary paper model is `bathyfacto`.

```bash
ns-train bathyfacto --data data/simulation-ship
```

For the no-refraction ablation used in the paper comparison, keep the same
method and disable the refraction branch:

```bash
ns-train bathyfacto --data data/simulation-ship --pipeline.model.disable-refraction True
```

The repository also keeps two Nerfacto comparison methods:

```bash
ns-train nerfacto-bathy-compare --data data/simulation-ship
ns-train nerfacto-bathy-compare-camopt --data data/simulation-ship
```

For like-for-like validation against the public release baseline, use a fixed
seed and explicitly disable camera optimization when comparing to a
camera-optimizer-off run:

```bash
ns-train bathyfacto \
  --data data/simulation-ship \
  --max-num-iterations 100000 \
  --machine.seed 42 \
  --pipeline.model.camera-optimizer.mode off
```

Replace `data/simulation-ship` only when using another compatible BathyFacto
dataset directory.
