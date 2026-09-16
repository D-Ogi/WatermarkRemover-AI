"""Pinned Florence assets with verified cache reuse and resumable downloads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from filelock import FileLock

from desktop_runtime import data_dir

MANIFEST = json.loads((Path(__file__).parent / "models/florence.json").read_text())
FLORENCE_REPO = MANIFEST["repo"]
FLORENCE_REVISION = MANIFEST["revision"]


def validate_endpoint(value):
    """Accept HTTPS base URLs without credentials, query strings or fragments."""
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or any(character.isspace() for character in value) or "\\" in value):
        raise ValueError("The model endpoint must be an HTTPS base URL without credentials, query or fragment.")
    # Validate explicit port syntax/range even if no download is needed yet.
    parsed.port
    return value.rstrip("/")


def save_endpoint(value):
    """Persist the installer's selection before downloading, including failed attempts."""
    endpoint = validate_endpoint(value)
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix="model-endpoint-", delete=False) as output:
            temporary = Path(output.name)
            json.dump({"endpoint": endpoint}, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, root / "model-endpoint.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return endpoint


def model_endpoint():
    """Use an explicit environment override, then the saved selection, then origin."""
    endpoint = os.environ.get("HF_ENDPOINT")
    if not endpoint:
        settings = data_dir() / "model-endpoint.json"
        endpoint = (json.loads(settings.read_text(encoding="utf-8"))["endpoint"]
                    if settings.exists() else "https://huggingface.co")
    return validate_endpoint(endpoint)


def florence_snapshot():
    """Reuse the standard Hugging Face snapshot directory for the pinned revision."""
    hf_home = Path(os.environ.get("HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "huggingface"))).expanduser()
    hub = Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))).expanduser()
    return hub / ("models--" + FLORENCE_REPO.replace("/", "--")) / "snapshots" / FLORENCE_REVISION


def valid_file(path, artifact):
    """Validate bytes before publishing or loading an artifact, including cached files."""
    try:
        if path.stat().st_size != artifact["size"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest() == artifact["sha256"]
    except FileNotFoundError:
        return False


def download_file(url, target, artifact, report):
    """Resume only a matching byte range; publish verified bytes atomically.

    Network interruptions retain partial bytes for retry. A checksum failure
    removes the partial file so retry cannot repeatedly reuse corrupt bytes.
    Callers hold the per-model file lock throughout validation and download.
    """
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= artifact["size"]:
        if valid_file(partial, artifact):
            os.replace(partial, target)
            return
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-", "Accept-Encoding": "identity"} if offset else {"Accept-Encoding": "identity"}
    deadline = time.monotonic() + 3600
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            expected = f"bytes {offset}-{artifact['size'] - 1}/{artifact['size']}"
            if content_range != expected:
                raise RuntimeError("Server returned an unexpected download range; retry the download.")
        elif response.status == 200:
            offset = 0  # Server ignored Range: restart instead of appending duplicate bytes.
        else:
            raise RuntimeError(f"Unexpected download response: {response.status}")
        report(offset, artifact["size"])
        with partial.open("ab" if offset else "wb") as output:
            while chunk := response.read(1024 * 1024):
                offset += len(chunk)
                if offset > artifact["size"] or time.monotonic() > deadline:
                    raise RuntimeError("Model download exceeded its expected size or time limit.")
                output.write(chunk)
                report(offset, artifact["size"])
            output.flush()
            os.fsync(output.fileno())
    if partial.stat().st_size < artifact["size"]:
        raise RuntimeError("Model transfer ended early. Please retry to resume the download.")
    if not valid_file(partial, artifact):
        partial.unlink(missing_ok=True)
        raise RuntimeError("Model download failed its SHA-256 check. Please retry.")
    os.replace(partial, target)


def ensure_florence(*, download=True, progress=None):
    """Return a complete verified pinned snapshot; offline checks never open the network."""
    download = download and os.environ.get("HF_HUB_OFFLINE", "").upper() not in {"1", "TRUE", "YES"}
    root = florence_snapshot()
    if download:
        root.mkdir(parents=True, exist_ok=True)
    def report(status, filename="", current=0, total=0):
        """Forward plain progress data to the caller without coupling to a GUI."""
        if progress:
            progress(dict(status=status, model="Florence-2", file=filename, current=current, total=total))
    def prepare():
        """Verify every required file and fetch only missing or damaged artifacts."""
        for artifact in MANIFEST["files"]:
            name = artifact["name"]
            target = root / name
            report("checking", name)
            if valid_file(target, artifact):
                continue
            if not download:
                raise RuntimeError(f"Florence-2 needs a download or repair: {name}")
            report("downloading", name, 0, artifact["size"])
            url = f"{model_endpoint()}/{FLORENCE_REPO}/resolve/{FLORENCE_REVISION}/{name}"
            download_file(url, target, artifact,
                          lambda current, total: report("downloading", name, current, total))
    if download:
        report("waiting")
        with FileLock(str(root / ".prepare.lock"), timeout=600):
            prepare()
    else:
        prepare()
    return root


def prepare_models(*, download=True, progress=None):
    """Verify and prepare both models before allowing desktop processing."""
    from lama_inpaint.model_store import ensure_model
    ensure_model(download=download, progress=progress)
    ensure_florence(download=download, progress=progress)


def main():
    """Emit progress as JSON lines for the isolated desktop preparation worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--florence-only", action="store_true")
    parser.add_argument("--endpoint", help="Persist an HTTPS Florence endpoint for future desktop retries")
    args = parser.parse_args()
    def emit(event):
        """Flush each event so the GUI can report progress while the worker runs."""
        print(json.dumps(event), flush=True)
    try:
        if args.endpoint is not None:
            save_endpoint(args.endpoint)
            os.environ["HF_ENDPOINT"] = args.endpoint
        prepare = ensure_florence if args.florence_only else prepare_models
        prepare(download=not args.check, progress=emit)
        emit(dict(status="ready", message="Models are verified and ready."))
    except Exception as exc:
        emit(dict(status="missing" if args.check else "error",
                  message="Model files are missing or need repair. Choose Download / Retry." if args.check else str(exc),
                  detail=str(exc)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
