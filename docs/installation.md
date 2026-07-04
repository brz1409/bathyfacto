# Installation

Use a Linux machine with an NVIDIA GPU and a CUDA-capable PyTorch installation.
The project follows the normal Nerfstudio editable-install workflow.

```bash
git clone <repository-url> bathyfacto
cd bathyfacto
pip install -e .
```

For accelerated hash-grid training, install `tiny-cuda-nn` in the same Python
environment following the upstream package instructions for your CUDA and
PyTorch versions.

After installation, verify that the public method is registered:

```bash
ns-train --help | grep bathyfacto
```
