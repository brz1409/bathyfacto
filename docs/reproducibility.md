# Reproducibility

This release is validated in two layers.

Download the dataset before running tests or training:

```bash
python scripts/download_dataset.py
```

First, an equivalence test checks that the model preserves the expected
forward path: it reproduces frozen reference scalars at `atol=1e-6` and emits
exactly the documented output keys.

```bash
python -m pytest tests/test_bathyfacto_equivalence.py -q
```

Second, a 100k-step training run of this package is checked against a
reference training run. Two reference number sets exist for the same model,
and they are not directly comparable:

- Published values (paper Table 2, eval split, masked valid pixels): PSNR
  34.18 dB, SSIM 0.893, LPIPS 0.095.
- `ns-eval` reference (full images, what the commands below reproduce):
  train PSNR 38.55 dB (from the training log), eval PSNR 36.08 dB, SSIM
  0.896, LPIPS 0.089.

The paper removes "no data" pixels before computing its metrics, while
`ns-eval` uses the full images, so the two sets of numbers differ. The four
tolerance bands below apply to the `ns-eval` reference:

- train PSNR within 0.5 dB
- eval PSNR within 0.5 dB
- eval SSIM within 0.01
- eval LPIPS within 0.01

Retraining this released code reproduced the ns-eval reference within all four tolerances.

```bash
ns-train bathyfacto \
    --data data/simulation-ship \
    --machine.seed 42 \
    --viewer.quit-on-train-completion True

ns-eval --load-config outputs/.../config.yml
```
