# Windows portable application

The portable package contains a versioned `WatermarkRemover-AI.exe`, CPython
3.13.15, the application dependencies, Qt WebEngine, and the complete offline
interface. No system Python, Git, administrator access or WebView2 installation
is needed. Extract the whole archive to a writable directory before opening the
EXE; do not launch it from inside the archive.

## First launch

1. Choose the CPU package for broad hardware compatibility, or the CUDA 12.6
   package for a compatible NVIDIA GPU/driver. See [device guidance](dependency-migration.md).
2. Extract the package and open `WatermarkRemover-AI.exe`.
3. Open **Models**, then choose **Download / Retry**. The application checks its
   pinned LaMA and Florence-2 files, downloads missing/damaged files and shows
   progress. Model data is about 1.8 GB; allow additional disk space for the
   application and temporary transfers.
4. After **ready** appears, choose input and output paths and process a file.
   Detection Preview uses the same prepared models.

A connection failure is shown in the Models panel. Fix the connection or free disk
space and use **Download / Retry** again. Completed verified files are reused.
Partial Florence-2 transfers resume when the server supports byte ranges; LaMA
restarts an interrupted transfer. No downloaded file is accepted before its pinned
size and SHA-256 match. Neither application updates nor arbitrary scripts are
fetched through this model downloader.

The UI, translations, themes and fonts work without internet. Processing uses
prepared local models. First model preparation requires access to GitHub and
Hugging Face, or a preseeded cache. See [offline cache instructions](lama-runtime.md).
The Florence-2 manifest and pinned snapshot revision are in `models/florence.json`.
Source installers remember the selected Florence mirror for later desktop retries.
An explicit `HF_ENDPOINT` takes precedence over the saved `model-endpoint.json` in
the application data directory; only HTTPS base URLs without credentials, query
strings or fragments are accepted. A mirror never changes the pinned checksums.
LaMA still uses its separate GitHub source. To persist a different Florence endpoint:

```sh
python -m model_assets --florence-only --endpoint https://huggingface.co
```

FFmpeg is still optional and must be available on PATH to preserve video audio.
The UI reports its availability. The package does not silently install a system
codec or driver. CUDA packages do not certify every NVIDIA card or older driver;
MPS is not a Windows feature and is outside the current LaMA device contract.

## Portable state and upgrades

`data/` beside the executable stores settings, logs and default model caches.
Explicit `TORCH_HOME`, `HF_HOME` or Hugging Face cache variables still take
precedence. Keep the `data` directory when moving the portable application.
For an upgrade, extract into a new directory, close the old application, then copy
`data/` into the new directory. Keep the previous version until the new one works.
Do not mix dependency folders from different versions or CPU/CUDA packages.

Source installations use the user's application data directory for new settings
and logs, and read an existing project `ui.yml` when no new settings file exists.
Existing theme and language choices are retained.

## Diagnostics

- `data/desktop.log` records windowed-launch errors. Inspect it before sharing;
  logs may contain local paths and processing details.
- `build-info.json` identifies version, backend, Python artifact and source commit.
- `installed-packages.txt` records the exact package versions included in the build.
- Launch `WatermarkRemover-AI.exe --check` to exercise the real local page and
  JavaScript/Python bridge without downloading production models. This is a UI
  installation check, not proof of GPU or model inference.
- For CLI usage, run `python/python.exe remwm.py INPUT OUTPUT` from the extracted
  directory. Keep cache variables consistent with your GUI cache; in a shell set
  `TORCH_HOME` to `data/models/torch` and `HF_HOME` to `data/models/huggingface`.

## Building and release checks

On Windows, from a clean checkout with a working build Python/pip:

```powershell
python scripts/build_windows.py --output dist --cache build-downloads --backend cpu
python scripts/build_windows.py --output dist --cache build-downloads --backend cu126
```

The builder verifies the embedded Python archive, uses the explicit Torch index,
checks dependency resolution and imports, compiles the native launcher and emits
versioned ZIP/SHA-256 files. Runtime DLLs and dependencies' license files remain
separate in the package. Python entry-point wrappers containing build-machine
paths are excluded; use `python.exe -m pip` for intentional package maintenance.
Never run installation commands into a release candidate after recording its
validation evidence. Build diagnostics are not a promise of byte-for-byte
reproducible archives: preserve the emitted package inventory and checksum.

For application-only rebuilds, `--refresh-app` requires matching Python, backend,
Torch/pip versions and dependency manifests from the previous build. It replaces
application directories so removed files cannot remain. Runtime changes or old
build metadata require a fresh output directory. User data and the embedded runtime
are preserved during an application-only refresh.

CI builds the CPU executable and exercises the actual offline desktop page.
The Windows portable workflow builds both CPU/CUDA artifacts on a version tag or
manual dispatch, without automatically publishing a release. Before publishing,
validate the extracted archive from another directory, first model download,
connection failure/retry, preview, processing, cancellation and error reporting.
Real model/GPU runs stay in a trusted environment outside pull-request CI.
