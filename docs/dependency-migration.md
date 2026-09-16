# Scientific runtime migration

This migration applies the updates proposed in #83, #84, #85, #86 and #87 to the
standalone LaMA application introduced in #88. Update the application as a unit:
OpenCV 5 requires NumPy 2 on the supported Python 3.12 environment. Raising only
the OpenCV requirement leaves an unsatisfiable dependency set.

| Component | Application requirement | Lightweight test version |
| --- | --- | --- |
| NumPy | `>=2,<3` | `2.5.3` |
| OpenCV headless | `>=5.0.0.93,<5.1.0` | `5.0.0.93` |
| PyTorch | `>=2.14.0` | Not installed |
| torchvision | `>=0.29.0` | Not installed |
| pywebview | `>=6.2.1`, with Qt extra on Linux | Not installed |
| Matplotlib | Not an application requirement | `3.11.2` |

## Install and device selection

Use a fresh Python 3.12 environment and the instructions in
[LaMA installation](lama-runtime.md#installation-and-migration). Run `pip check`
in the application environment after installation; the separate test environment
cannot establish application dependency compatibility.

Linux setup selects the official CPU index without NVIDIA, or CUDA 12.6 with
NVIDIA. The older CUDA 12.4 index does not contain torch 2.14. The CUDA 12.6 index
provides the matching torch 2.14 / torchvision 0.29 pair. These commands ignore
inherited pip environment options and user configuration. The package mirror
still applies to other application dependencies.

For a manual CUDA 12.6 installation on supported Windows/Linux hardware, install
the matching pair before the application requirements:

```sh
python -m pip --isolated install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.ones(1, device='cuda').cpu())"
```

Use the `cpu` index instead of `cu126` for an explicit CPU installation and omit
the CUDA assertion. macOS uses its native PyPI wheels. Windows setup continues to
install from PyPI/the selected package mirror; this migration does not add
hardware detection to the Windows installers. A generic PyPI installation is
not a guarantee of a CUDA build.

An installed CUDA toolkit is not a substitute for a compatible NVIDIA driver and
GPU architecture. Consult [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
and [official PyTorch builds](https://download.pytorch.org/whl/cu126/torch/).
CUDA 12.x minor compatibility has feature/PTX limitations on older drivers.
Hardware requiring another build needs an explicitly selected compatible pair.
MPS inference remains outside the validated LaMA device contract.

## Validation scope

CI installs both manifests on Windows/Linux/macOS and checks dependency
resolution, generated TorchScript inference, image/video routing, and a real
JavaScript/Python desktop bridge. The lightweight environment now uses the same
OpenCV major version as the application. Shell cases check CPU/CUDA index
selection, mirror independence and failure propagation.

Real model validation must additionally compare CPU output with the previous
NumPy/OpenCV environment, run both video paths, and exercise Florence-2 and a
named CUDA device. Report concrete results in the pull request; do not infer
GPU or real-model compatibility from the generated fixture alone.
