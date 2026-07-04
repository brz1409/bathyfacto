# Point-Cloud Export

After training, export a refractive point cloud with the Bathy point-cloud
exporter.

```bash
ns-export bathy-pointcloud \
  --load-config <RUN_DIR>/config.yml \
  --output-dir <EXPORT_DIR>
```

Use the `config.yml` written by the training run. The exporter reloads the
trained model, uses the stored dataparser transform metadata, and writes the
exported geometry into the requested output directory.

The exact export settings depend on the dataset and the desired point density.
Keep export commands and generated point clouds outside the repository.
