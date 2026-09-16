# Standalone LaMA runtime

The application loads the existing `big-lama.pt` TorchScript model directly.
IOPaint, its model registry/CLI, diffusion backends, and request schemas are no
longer needed. Florence-2 detection, desktop themes and translations retain their
existing behavior.

## Installation and migration

Use a fresh application environment when upgrading an installation that contains
IOPaint. Leaving that package installed can retain conflicting requirements even
though the application no longer imports it. The installers do not remove packages
from another environment and do not suppress dependency failures.

For a manual installation with Python 3.12:

```sh
python -m venv .venv-app
# Linux/macOS: . .venv-app/bin/activate
# Windows PowerShell: .venv-app\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -m lama_inpaint download
python remwm.py --help
```

For a headless CLI installation, use `requirements-core.txt` instead. Select a
matching torch/torchvision build for your device before installing requirements;
the generic manifest does not guarantee a CUDA build. Linux desktop installation
uses pywebview's Qt extra, Windows keeps its native backend, and macOS keeps its
native backend. Linux still needs a graphical session and the Qt system libraries. On Ubuntu
24.04, prepare the desktop libraries with:

```sh
sudo apt-get install libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xinerama0 libxcb-randr0 libnss3 libasound2t64
```

Package names may differ on other distributions. The installer does not silently
install system packages or elevate privileges.

The standalone runtime removes the IOPaint `imghdr` import path. This does not by
itself certify every dependency or installer on Python 3.13+. The application CI
currently validates Python 3.12.

## Model integrity and offline use

All three installers and runtime loading use the same model store. It reuses
`$TORCH_HOME/hub/checkpoints/big-lama.pt`, or otherwise
`$XDG_CACHE_HOME/torch/hub/checkpoints/big-lama.pt` (default `~/.cache/torch/...`).
An existing valid IOPaint cache is reused without downloading it again. A custom
`torch.hub.set_dir()` made by unrelated Python code is not consulted.

```sh
python -m lama_inpaint download
python -m lama_inpaint download --offline
python -m lama_inpaint download --cache-dir /path/to/checkpoints
```

`--cache-dir` applies to that preparation command; set `TORCH_HOME` for the normal
application to use a nondefault cache. Offline verification fails if the file is
missing or corrupt and never opens a network connection. Programmatic callers
can use `LamaInpaint(device="cpu", download=False)` for strict offline loading.

The artifact URL, exact byte count and SHA-256 are pinned in
`lama_inpaint/model_store.py`; see [provenance](../THIRD_PARTY_NOTICES.md).
Cached weights are verified before deserialization. Downloads have read/time/size
limits, a cross-process lock, a unique temporary file and atomic replacement after
verification. Interrupted or invalid downloads never become the cached model and
do not delete an existing artifact. An invalid cache is replaced only after a new
download passes verification. No user-provided URL or checksum environment variable
can bypass the pinned artifact check.

## Restricted networks

The China option changes the package and Florence-2 mirrors. LaMA still uses the
pinned GitHub artifact; there is no unverified model mirror or URL override.
If GitHub is unavailable, preseed the cache before running setup:

1. On a machine with GitHub access and these application packages installed, run
   `python -m lama_inpaint download --cache-dir model-transfer`.
2. Transfer `model-transfer/big-lama.pt` to the target machine. Place it at
   `TORCH_HOME/hub/checkpoints/big-lama.pt` if you set `TORCH_HOME`, otherwise at
   `~/.cache/torch/hub/checkpoints/big-lama.pt`. If `XDG_CACHE_HOME` is set, use
   `XDG_CACHE_HOME/torch/hub/checkpoints/big-lama.pt` instead of `~/.cache/...`.
   Keep the cache environment variables consistent when running setup and the app.
3. Run setup normally with the chosen package mirror. Every installer checks the
   local artifact before attempting any model network request. Valid preseeded
   weights are reused; missing or incorrect weights still cause setup to fail if
   GitHub cannot be reached. Do not bypass the checksum check.
4. In the installed application environment, confirm the cache without network
   access using `python -m lama_inpaint download --offline`.

Linux installs CPU and CUDA PyTorch builds from their respective official wheel
indexes, including when a different mirror is selected for other packages. The
CUDA branch currently selects CUDA 12.4; this is not a promise of support for every
GPU/driver generation. Hardware requiring another build still needs an explicitly
selected compatible torch/torchvision pair.

## Processing contract

- Input: nonempty `uint8` RGB array `(H, W, 3)` and matching `uint8` mask `(H, W)`.
- Output: independent `uint8` BGR array, preserving the application's existing API.
- Values above zero select model pixels; mask values divided by 255 determine the
  final composite. Zero-mask pixels are preserved exactly.
- Images above 800 pixels on the longest side use connected-region crops with the
  existing 64-pixel context margin. Crops are predicted from the original image,
  including overlapping context regions.
- Symmetric padding aligns dimensions to multiples of 8. Tiny inputs are padded
  to at least 32 pixels per dimension, then cropped back; they are never resized.
- Empty masks bypass inference. Caller-owned image/mask arrays are never modified.
- CPU and CUDA are supported. An unavailable requested CUDA device fails explicitly;
  MPS is rejected until its operators and results are validated.

The old `resize_limit=1600` belonged to an unused RESIZE strategy; the application
selected CROP. It was not a hard cap on a full-image mask. The new adapter likewise
does not resize large masks silently. Memory requirements still depend on crop size.

Intentional differences from the previous wrapper: no input mutation, no model
call for an empty mask, safe padding for tiny images, and inclusion of soft masks
below 128 when choosing large-image crops. Standard binary-mask processing retains
the previous crop/pad/composite behavior.

## Validation

The lightweight suite exercises processing and download integrity without PyTorch
or real model weights. The separate application job installs the complete manifest
on Windows/Linux/macOS, runs `pip check`, checks imports/CLI help, and executes a
small generated TorchScript model plus image/video routing tests. Neither job
downloads production weights or certifies GPU execution.

Changes to model execution also need a separately recorded real-weight comparison:
model digest, runtime versions, device, fixture dimensions/masks and differences
from the reference. Keep fixtures synthetic or redistributable. GPU coverage must
name the tested device; it does not imply universal GPU or MPS compatibility.
