# Installation

Use a Linux machine with an NVIDIA GPU and a CUDA-capable PyTorch installation.
Tested with Python 3.10, torch 2.2.2+cu118, torchvision 0.17.2, and
tiny-cuda-nn.

From a clone of this repository:

```bash
git clone https://github.com/brz1409/bathyfacto.git && cd bathyfacto
```

```bash
conda create -n bathyfacto python=3.10 -y && conda activate bathyfacto
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install "setuptools<81" wheel ninja
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e .
python scripts/download_dataset.py
ns-train bathyfacto --data data/simulation-ship
```

`tiny-cuda-nn` provides the accelerated hash-grid encoding the paper model
uses. Build it in the same Python environment as the rest of the project,
matching your CUDA and PyTorch versions. Its `setup.py` needs `pkg_resources`
(setuptools 81 dropped it) and expects `torch` already importable, hence the
`setuptools<81` pin and `--no-build-isolation` above. If the build cannot find
`nvcc`, set `CUDA_HOME` to your CUDA 11.8 install and put `$CUDA_HOME/bin` on
`PATH` before the `pip install --no-build-isolation` line. `TCNN_CUDA_ARCHITECTURES`
can be set to your GPU's compute capability (e.g. `89` for an L40S, `80` for
an A100) to skip building for architectures you do not need.

After installation, verify that the public method is registered:

```bash
ns-train --help | grep bathyfacto
```
