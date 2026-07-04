# Reproducibility

The public release is validated in two layers.

Download the dataset before running tests or training:

```bash
python scripts/download_dataset.py
```

First, an equivalence test checks that the excised paper-only model preserves
the expected forward path and does not emit removed research output keys.

```bash
python -m pytest tests/models/test_bathyfacto_equivalence.py -q
```

Second, a 100k-step confirming training run of the public package is compared
with baseline run `1xzzttp8`. The comparison uses camera optimization disabled,
seed 42, and five tolerance bands:

- train PSNR within 0.5 dB
- eval PSNR within 0.5 dB
- eval SSIM within 0.01
- eval LPIPS within 0.01
- depth-bias median within 0.01 m

The release should not be treated as complete until all five bands pass and the
result has been reviewed.
