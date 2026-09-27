# Training

The primary paper model is `bathyfacto`. Camera optimization is off by
default, matching the paper setup.

```bash
ns-train bathyfacto --data data/simulation-ship
```

For the no-refraction ablation used in the paper comparison, keep the same
method and disable the refraction branch:

```bash
ns-train bathyfacto --data data/simulation-ship --pipeline.model.disable-refraction True
```

The repository also includes a Nerfacto comparison method, likewise with
camera optimization off:

```bash
ns-train nerfacto-bathy-compare --data data/simulation-ship
```

For like-for-like validation against the reference run, use a fixed seed:

```bash
ns-train bathyfacto --data data/simulation-ship --machine.seed 42
```

Replace `data/simulation-ship` only when using another compatible BathyFacto
dataset directory.
