"""Verified, atomic download and reuse of the fixed LaMA TorchScript artifact."""

import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

from filelock import FileLock, Timeout

MODEL_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
)
MODEL_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
MODEL_SIZE = 205669692
MODEL_NAME = "big-lama.pt"
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_DEADLINE = 600
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 1.0


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


def ensure_model(cache_dir=None, *, download=True, progress=None, verbose=False):
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
            _download(target, progress=progress, verbose=verbose)
    except Timeout as exc:
        raise ModelError("Timed out waiting for another LaMA model download.") from exc
    return target


def _format_bytes(size):
    """Format a byte count for human-readable progress output."""
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024


def _format_duration(seconds):
    """Format a duration without requiring a third-party progress package."""
    if seconds <= 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _verbose(verbose, message):
    """Write downloader diagnostics immediately when verbose mode is enabled."""
    if verbose:
        print(f"[LaMA] {message}", file=sys.stderr, flush=True)


def _response_status(response):
    """Read an HTTP status while remaining compatible with simple test streams."""
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        status = getcode() if getcode is not None else None
    return status


def _response_header(response, name):
    """Read a response header from real HTTP responses or return None for test streams."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    return headers.get(name)


def _download(target, progress=None, *, verbose=False):
    """Download, resume and verify the model before publishing it atomically."""
    partial = target.with_name(f".{target.name}.part")
    started_at = time.monotonic()
    last_progress = 0.0

    try:
        existing = partial.stat().st_size if partial.exists() else 0
        if existing > MODEL_SIZE:
            _verbose(verbose, f"Partial file exceeds expected size; restarting: {partial}")
            partial.unlink()
            existing = 0

        if existing == MODEL_SIZE:
            _verbose(verbose, "Partial file is complete; verifying SHA-256...")
            if _is_valid(partial):
                os.replace(partial, target)
                _verbose(verbose, "Partial file verified and published as the model.")
                return
            _verbose(verbose, "Complete partial file failed SHA-256; restarting.")
            partial.unlink()
            existing = 0

        _verbose(
            verbose,
            f"Target: {target} | Partial: {partial} | "
            f"Expected size: {_format_bytes(MODEL_SIZE)}",
        )

        response = None
        resumed = existing > 0
        if resumed:
            _verbose(
                verbose,
                f"Resuming from {_format_bytes(existing)} "
                f"({existing / MODEL_SIZE:.1%}) using HTTP Range...",
            )
            request = Request(MODEL_URL, headers={"Range": f"bytes={existing}-"})
            response = urlopen(request, timeout=DOWNLOAD_TIMEOUT)
            status = _response_status(response)
            if status != 206:
                _verbose(
                    verbose,
                    f"Server did not accept resume (HTTP {status or 'unknown'}); "
                    "restarting the transfer.",
                )
                response.close()
                response = None
                partial.unlink()
                existing = 0
                resumed = False

        if response is None:
            _verbose(verbose, f"Downloading from {MODEL_URL}")
            response = urlopen(MODEL_URL, timeout=DOWNLOAD_TIMEOUT)

        content_range = _response_header(response, "Content-Range")
        if resumed and content_range:
            expected_prefix = f"bytes {existing}-"
            if not content_range.startswith(expected_prefix):
                response.close()
                raise ModelError(
                    f"Server returned an unexpected range ({content_range}); "
                    f"partial file retained at {partial} for retry."
                )

        mode = "ab" if existing else "wb"
        downloaded = existing
        previous_downloaded = downloaded
        progress_started_at = time.monotonic()
        last_progress = 0.0
        _verbose(
            verbose,
            f"Transfer {'resumed' if resumed else 'started'}; "
            "interrupted transfers will retain their progress.",
        )

        with response:
            with partial.open(mode) as output:
                while True:
                    if time.monotonic() - started_at > DOWNLOAD_DEADLINE:
                        raise ModelError(
                            f"Download deadline reached at {_format_bytes(downloaded)}/"
                            f"{_format_bytes(MODEL_SIZE)}; partial file retained at {partial}."
                        )
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > MODEL_SIZE:
                        raise ModelError(
                            f"Download exceeds expected size "
                            f"({_format_bytes(downloaded)} > {_format_bytes(MODEL_SIZE)}); "
                            f"partial file retained at {partial}."
                        )
                    now = time.monotonic()
                    if verbose and (
                        now - last_progress >= PROGRESS_INTERVAL
                        or downloaded >= MODEL_SIZE
                    ):
                        elapsed = max(now - progress_started_at, 0.001)
                        speed = (downloaded - previous_downloaded) / elapsed
                        remaining = max(MODEL_SIZE - downloaded, 0)
                        eta = remaining / speed if speed > 0 else float("inf")
                        print(
                            f"\r[LaMA] {_format_bytes(downloaded)}/"
                            f"{_format_bytes(MODEL_SIZE)} "
                            f"({downloaded / MODEL_SIZE:.1%}) | "
                            f"{_format_bytes(speed)}/s | ETA {_format_duration(eta)}",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_progress = now
                    if progress:
                        progress(
                            dict(
                                status="downloading",
                                model="LaMA",
                                current=downloaded,
                                total=MODEL_SIZE,
                            )
                        )
                output.flush()
                os.fsync(output.fileno())

        if verbose:
            print(file=sys.stderr, flush=True)
        if downloaded != MODEL_SIZE:
            raise ModelError(
                f"Incomplete download: {_format_bytes(downloaded)}/"
                f"{_format_bytes(MODEL_SIZE)}; partial file retained at {partial}."
            )

        _verbose(verbose, "Download complete; verifying size and SHA-256...")
        with partial.open("rb") as stream:
            verify_stream(stream)
        os.replace(partial, target)
        _verbose(verbose, f"Model verified and saved to {target}")
    except ModelError:
        raise
    except (OSError, ValueError, HTTPException) as exc:
        raise ModelError(
            f"Could not download LaMA: {exc}. "
            f"Partial file retained at {partial}; run the command again."
        ) from exc
