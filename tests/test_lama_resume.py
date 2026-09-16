"""Exercise resumable transfer boundaries without downloading model weights."""

import hashlib
import io
from urllib.error import HTTPError

import pytest

from lama_inpaint import model_store as store


class Response(io.BytesIO):
    """HTTP-like stream whose closure and headers can be checked."""

    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    """Use a small pinned artifact and retain an existing invalid cache file."""
    data = b"controlled model fixture"
    monkeypatch.setattr(store, "MODEL_SIZE", len(data))
    monkeypatch.setattr(store, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    target = tmp_path / store.MODEL_NAME
    target.write_bytes(b"old cache")
    partial = tmp_path / f".{store.MODEL_NAME}.part"
    partial.write_bytes(data[:5])
    return data, target, partial


@pytest.mark.parametrize("content_range", [
    None, "bytes 0-23/24", "bytes 5-23/99", "bytes 5-999/24",
    "bytes 5-3/24", "bytes 5-garbage", "bytes 5-23/*",
])
def test_invalid_range_preserves_partial(transfer, monkeypatch, content_range):
    """Never append a response whose range does not describe the pinned artifact."""
    data, target, partial = transfer
    headers = {} if content_range is None else {"Content-Range": content_range}
    response = Response(data[5:], 206, headers)
    monkeypatch.setattr(store, "urlopen", lambda *a, **k: response)
    with pytest.raises(store.ModelError, match="range"):
        store.ensure_model(target.parent)
    assert partial.read_bytes() == data[:5]
    assert target.read_bytes() == b"old cache"
    assert response.closed


def test_server_ignoring_range_reuses_full_response(transfer, monkeypatch):
    """A 200 response restarts safely with one request, without appending old bytes."""
    data, target, partial = transfer
    calls = []
    def open_response(request, **kwargs):
        calls.append(request)
        return Response(data, headers={"Content-Length": str(len(data))})
    monkeypatch.setattr(store, "urlopen", open_response)
    assert store.ensure_model(target.parent).read_bytes() == data
    assert len(calls) == 1 and not partial.exists()


def test_range_not_satisfiable_retries_from_zero(transfer, monkeypatch):
    """An HTTP 416 permits a clean full request without accepting unchecked bytes."""
    data, target, partial = transfer
    calls = []
    error_stream = io.BytesIO()
    def open_response(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(store.MODEL_URL, 416, "range rejected", {}, error_stream)
        return Response(data)
    monkeypatch.setattr(store, "urlopen", open_response)
    assert store.ensure_model(target.parent).read_bytes() == data
    assert len(calls) == 2 and error_stream.closed and not partial.exists()


@pytest.mark.parametrize("headers", [
    {"Content-Length": "garbage"}, {"Content-Length": "999"},
    {"Content-Length": "0"}, {"Content-Encoding": "gzip"},
])
def test_invalid_full_headers_keep_partial(transfer, monkeypatch, headers):
    """Validate a full response before truncating previously downloaded bytes."""
    data, target, partial = transfer
    response = Response(data, headers=headers)
    monkeypatch.setattr(store, "urlopen", lambda *a, **k: response)
    with pytest.raises(store.ModelError):
        store.ensure_model(target.parent)
    assert partial.read_bytes() == data[:5]
    assert target.read_bytes() == b"old cache" and response.closed


def test_oversized_chunk_not_written(transfer, monkeypatch):
    """Reject excessive bytes before writing them or replacing the old artifact."""
    data, target, partial = transfer
    response = Response(data[5:] + b"unexpected", 206,
                        {"Content-Range": f"bytes 5-{len(data)-1}/{len(data)}"})
    monkeypatch.setattr(store, "urlopen", lambda *a, **k: response)
    with pytest.raises(store.ModelError, match="size"):
        store.ensure_model(target.parent)
    assert partial.read_bytes() == data[:5]
    assert target.read_bytes() == b"old cache"


def test_deadline_during_last_read_prevents_publication(transfer, monkeypatch):
    """A final read that crosses the deadline must not publish a model."""
    data, target, partial = transfer
    now = [0.0]
    monkeypatch.setattr(store.time, "monotonic", lambda: now[0])
    class LateResponse(Response):
        def read(self, size=-1):
            chunk = super().read(size)
            if not chunk:
                now[0] = store.DOWNLOAD_DEADLINE + 1
            return chunk
    response = LateResponse(data)
    monkeypatch.setattr(store, "urlopen", lambda *a, **k: response)
    with pytest.raises(store.ModelError, match="deadline"):
        store.ensure_model(target.parent)
    assert target.read_bytes() == b"old cache"


@pytest.mark.parametrize("partial_data", [b"complete", b"corrupt", b"oversized"])
def test_complete_partial_reuse_or_repair(transfer, monkeypatch, partial_data):
    """Publish a verified complete partial offline; redownload unusable partials."""
    data, target, partial = transfer
    partial.write_bytes(data if partial_data == b"complete" else
                        b"x" * (len(data) + (partial_data == b"oversized")))
    calls = []
    def open_response(*args, **kwargs):
        calls.append(1)
        return Response(data)
    monkeypatch.setattr(store, "urlopen", open_response)
    assert store.ensure_model(target.parent).read_bytes() == data
    assert len(calls) == (partial_data != b"complete")
    assert not partial.exists()


def test_verbose_resume_keeps_structured_progress(transfer, monkeypatch, capsys):
    """Human progress goes to stderr while callbacks still report absolute bytes."""
    data, target, partial = transfer
    response = Response(data[5:], 206,
                        {"Content-Range": f"bytes 5-{len(data)-1}/{len(data)}"})
    monkeypatch.setattr(store, "urlopen", lambda *a, **k: response)
    events = []
    store.ensure_model(target.parent, verbose=True, progress=events.append)
    captured = capsys.readouterr()
    assert not captured.out
    assert "Resuming" in captured.err and "ETA" in captured.err and "SHA-256" in captured.err
    assert events[-1]["current"] == len(data) == events[-1]["total"]


@pytest.mark.parametrize("resume_policy", ["range", "ignore", "reject"])
def test_interrupted_real_http_download_resumes(tmp_path, monkeypatch, resume_policy):
    """Exercise urllib and real HTTP framing across interrupted and resumed calls."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    data = b"controlled model fixture"
    monkeypatch.setattr(store, "MODEL_SIZE", len(data))
    monkeypatch.setattr(store, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            """Keep local fixture requests out of test output."""

        def do_GET(self):
            """Drop the first response, then follow the selected range policy."""
            requested_range = self.headers.get("Range")
            requests.append(requested_range)
            if len(requests) == 1:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data[:9])
                self.wfile.flush()
                self.close_connection = True
                return
            if requested_range and resume_policy == "reject":
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start = 9 if requested_range and resume_policy == "range" else 0
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
            self.send_header("Content-Length", str(len(data)-start))
            self.end_headers()
            self.wfile.write(data[start:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(store, "MODEL_URL", f"http://127.0.0.1:{server.server_port}/model")
    try:
        with pytest.raises(store.ModelError, match="Incomplete"):
            store.ensure_model(tmp_path)
        partial = tmp_path / f".{store.MODEL_NAME}.part"
        assert partial.read_bytes() == data[:9]
        assert store.ensure_model(tmp_path).read_bytes() == data
        assert requests == [None, "bytes=9-"] + ([None] if resume_policy == "reject" else [])
        assert not partial.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_transient_http_error_preserves_partial(transfer, monkeypatch):
    """A service failure retains progress and closes the HTTP error response."""
    data, target, partial = transfer
    error_stream = io.BytesIO()
    def unavailable(*args, **kwargs):
        raise HTTPError(store.MODEL_URL, 503, "unavailable", {}, error_stream)
    monkeypatch.setattr(store, "urlopen", unavailable)
    with pytest.raises(store.ModelError, match="503"):
        store.ensure_model(target.parent)
    assert error_stream.closed and partial.read_bytes() == data[:5]
    assert target.read_bytes() == b"old cache"



def test_initial_http_error_closes_response(tmp_path, monkeypatch):
    """A failed initial request closes its response and creates no partial file."""
    error_stream = io.BytesIO()
    def unavailable(*args, **kwargs):
        raise HTTPError(store.MODEL_URL, 503, "unavailable", {}, error_stream)
    monkeypatch.setattr(store, "urlopen", unavailable)
    with pytest.raises(store.ModelError, match="503"):
        store.ensure_model(tmp_path)
    assert error_stream.closed
    assert not (tmp_path / f".{store.MODEL_NAME}.part").exists()
