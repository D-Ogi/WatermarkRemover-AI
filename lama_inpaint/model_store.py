"""Verified, atomic download and reuse of the fixed LaMA TorchScript artifact."""

import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import tempfile
import time
from urllib.request import urlopen

from filelock import FileLock, Timeout

MODEL_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
)
MODEL_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
MODEL_SIZE = 205669692
MODEL_NAME = "big-lama.pt"


class ModelError(RuntimeError):
    """A model could not be obtained or verified; no unverified weights are loaded."""


def default_cache_dir():
    """Honor the standard Torch cache variables and reuse existing LaMA weights."""
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return Path(torch_home).expanduser() / "hub" / "checkpoints"
    cache = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache / "torch" / "hub" / "checkpoints"


def verify_stream(stream):
    """Check an open artifact's exact size and SHA-256, then rewind it."""
    stream.seek(0)
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        size += len(chunk)
        if size > MODEL_SIZE:
            raise ModelError("LaMA model exceeds the expected artifact size.")
        digest.update(chunk)
    stream.seek(0)
    if size != MODEL_SIZE or digest.hexdigest() != MODEL_SHA256:
        raise ModelError("LaMA model checksum or size mismatch; refusing to load it.")


def _is_valid(path):
    """Return whether an existing artifact matches the pinned size and digest."""
    try:
        with path.open("rb") as stream:
            verify_stream(stream)
        return True
    except (FileNotFoundError, ModelError):
        return False


def ensure_model(cache_dir=None, *, download=True, progress=None):
    """Return verified weights, downloading atomically when allowed.

    Downloads use a unique temporary file and a per-artifact cross-process lock.
    A failed transfer never replaces an existing file. Offline mode never opens
    a network connection. The URL and expected digest are deliberately fixed.
    """
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    target = cache / MODEL_NAME
    if progress:
        progress(dict(status="checking", model="LaMA", current=0, total=MODEL_SIZE))
    if not download:
        if not _is_valid(target):
            raise ModelError(
                f"No verified LaMA model at {target}. Run python -m lama_inpaint download first."
            )
        return target
    cache.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(target) + ".lock", timeout=600):
            if _is_valid(target):
                return target
            _download(target, progress=progress)
    except Timeout as exc:
        raise ModelError("Timed out waiting for another LaMA model download.") from exc
    return target


def _download(target, progress=None):
    """Publish only a complete verified transfer; retain the old file on failure."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".lama-", suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            deadline = time.monotonic() + 600
            size = 0
            with urlopen(MODEL_URL, timeout=30) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MODEL_SIZE or time.monotonic() > deadline:
                        raise ModelError(
                            "LaMA download exceeded its size or time limit."
                        )
                    output.write(chunk)
                    if progress:
                        progress(dict(status="downloading", model="LaMA", current=size, total=MODEL_SIZE))
            output.flush()
            os.fsync(output.fileno())
        with temporary.open("rb") as stream:
            verify_stream(stream)
        os.replace(temporary, target)
    except (OSError, ValueError, HTTPException) as exc:
        raise ModelError(
            f"Could not download LaMA weights: {exc}. Retry python -m lama_inpaint download."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
