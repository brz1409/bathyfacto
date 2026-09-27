# Point-cloud export

After training, export a refraction-corrected point cloud with the BathyFacto
exporter.

```bash
bathyfacto-export --load-config <RUN_DIR>/config.yml
```

Use the `config.yml` written by the training run. The exporter reloads the
trained model, uses the stored dataparser transform metadata, and writes the
exported geometry into the requested output directory (default
`<RUN_DIR>/exports/bathy-pointcloud`). Pass `--output-dir` to choose another
location and `--num-points` to change the maximum number of exported points
(default 1,000,000).

Keep export commands and generated point clouds outside the repository.
