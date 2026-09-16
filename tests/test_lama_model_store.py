"""Download integrity and failure behavior using tiny local byte streams."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path

import pytest

from lama_inpaint import model_store as store


class Response(io.BytesIO):
    """A controlled successful HTTP response with an optional header mapping."""

    status = 200
    headers = {}


@pytest.fixture
def artifact(monkeypatch):
    """Pin verification to tiny controlled bytes so download tests need no real weights."""
    data = b"controlled model fixture"
    monkeypatch.setattr(store, "MODEL_SIZE", len(data))
    monkeypatch.setattr(store, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    return data


def test_download_then_offline_cache_reuse(tmp_path, monkeypatch, artifact):
    """Download once and verify that both online and offline reuse avoid another request."""
    calls = []

    def response(url, timeout):
        """Record a network attempt and supply the controlled artifact bytes."""
        assert url == store.MODEL_URL and timeout == 30
        calls.append(url)
        return Response(artifact)

    monkeypatch.setattr(store, "urlopen", response)
    target = store.ensure_model(tmp_path)
    assert target.read_bytes() == artifact
    assert store.ensure_model(tmp_path, download=False) == target
    assert store.ensure_model(tmp_path) == target
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("bad", [b"", b"truncated", b"x" * 200])
def test_bad_download_never_published(tmp_path, monkeypatch, artifact, bad):
    """Reject bad transfers without publishing them, while retaining resumable data."""
    monkeypatch.setattr(store, "urlopen", lambda *a, **k: Response(bad))
    with pytest.raises(store.ModelError):
        store.ensure_model(tmp_path)
    assert not (tmp_path / store.MODEL_NAME).exists()
    assert (tmp_path / f".{store.MODEL_NAME}.part").exists()


def test_same_size_tampering_rejected_offline(tmp_path, artifact):
    """Require the digest as well as size when accepting offline weights."""
    (tmp_path / store.MODEL_NAME).write_bytes(b"x" * len(artifact))
    with pytest.raises(store.ModelError, match="No verified"):
        store.ensure_model(tmp_path, download=False)


def test_missing_offline_model_does_not_connect(tmp_path, monkeypatch, artifact):
    """Fail an absent offline cache without opening a network connection."""

    def fail(*a, **k):
        """Reject an unexpected network request in an offline test."""
        pytest.fail("offline mode must not connect")

    monkeypatch.setattr(store, "urlopen", fail)
    with pytest.raises(store.ModelError):
        store.ensure_model(tmp_path, download=False)


def test_interrupted_download_preserves_old_file(tmp_path, monkeypatch, artifact):
    """Keep the old artifact and resumable data after a connection interruption."""
    old = tmp_path / store.MODEL_NAME
    old.write_bytes(b"old invalid file")

    class Interrupted(Response):
        def read(self, size=-1):
            """Supply one partial chunk, then simulate an interrupted connection."""
            if self.tell():
                raise OSError("connection interrupted")
            return super().read(5)

    monkeypatch.setattr(store, "urlopen", lambda *a, **k: Interrupted(artifact))
    with pytest.raises(store.ModelError, match="interrupted"):
        store.ensure_model(tmp_path)
    assert old.read_bytes() == b"old invalid file"
    assert (tmp_path / f".{store.MODEL_NAME}.part").read_bytes() == artifact[:5]


def test_interrupted_download_resumes_with_http_range(tmp_path, monkeypatch, artifact):
    """Resume a retained partial artifact instead of downloading it from zero."""
    partial = tmp_path / f".{store.MODEL_NAME}.part"
    partial.write_bytes(artifact[:5])

    class Resumed(Response):
        status = 206
        headers = {"Content-Range": f"bytes 5-{len(artifact) - 1}/{len(artifact)}"}

    def response(request, timeout):
        assert request.headers["Range"] == "bytes=5-"
        return Resumed(artifact[5:])

    monkeypatch.setattr(store, "urlopen", response)
    target = store.ensure_model(tmp_path)
    assert target.read_bytes() == artifact
    assert not partial.exists()


def test_corrupt_cache_replaced_only_after_verification(
    tmp_path, monkeypatch, artifact
):
    """Replace an invalid existing artifact only with the complete verified fixture."""
    target = tmp_path / store.MODEL_NAME
    target.write_bytes(b"bad")
    monkeypatch.setattr(store, "urlopen", lambda *a, **k: Response(artifact))
    assert store.ensure_model(tmp_path).read_bytes() == artifact


def test_concurrent_downloads_share_verified_artifact(tmp_path, monkeypatch, artifact):
    """Concurrent callers must obtain one verified download through the artifact lock."""
    calls = []

    def response(*a, **k):
        """Record a network attempt and supply the controlled artifact bytes."""
        calls.append(1)
        return Response(artifact)

    monkeypatch.setattr(store, "urlopen", response)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: store.ensure_model(tmp_path), range(4)))
    assert all(p.read_bytes() == artifact for p in paths)
    assert len(calls) == 1


def test_torch_cache_environment(monkeypatch, tmp_path):
    """Apply Torch cache precedence before falling back to the XDG cache directory."""
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "custom"))
    assert store.default_cache_dir() == tmp_path / "custom" / "hub" / "checkpoints"
    monkeypatch.delenv("TORCH_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert store.default_cache_dir() == tmp_path / "torch" / "hub" / "checkpoints"


def test_verification_rewinds_stream(artifact):
    """Leave a verified open stream ready for deserialization from its beginning."""
    stream = Response(artifact)
    stream.seek(4)
    store.verify_stream(stream)
    assert stream.tell() == 0


def test_incomplete_http_response_is_reported_and_cleaned(
    tmp_path, monkeypatch, artifact
):
    """Convert an incomplete HTTP response into a controlled error without residue."""
    from http.client import IncompleteRead

    def interrupted(*args, **kwargs):
        """Simulate an incomplete HTTP response before an artifact can be published."""
        raise IncompleteRead(b"partial", len(artifact))

    monkeypatch.setattr(store, "urlopen", interrupted)
    with pytest.raises(store.ModelError, match="Could not download"):
        store.ensure_model(tmp_path)
    assert not (tmp_path / store.MODEL_NAME).exists()
    assert not list(tmp_path.glob("*.part"))
