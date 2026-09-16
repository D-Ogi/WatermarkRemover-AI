"""Verify progress, interruption, safe resumption and pinned model integrity."""
import hashlib
import io

import pytest
import model_assets as assets


@pytest.fixture
def artifact():
    """Use tiny fixed bytes instead of real model downloads."""
    data = b"verified model fixture"
    return data, dict(name="model.bin", size=len(data), sha256=hashlib.sha256(data).hexdigest())


class Response(io.BytesIO):
    """Minimal HTTP stream with explicit status and range headers."""
    status = 200
    headers = {}


def test_interruption_and_resume_publish_only_verified_file(tmp_path, monkeypatch, artifact):
    """Keep a partial transfer after a network error and resume at its exact offset."""
    data, info = artifact
    target = tmp_path / info['name']
    class Interrupted(Response):
        def read(self, size=-1):
            """Return one chunk before simulating a lost connection."""
            if self.tell():
                raise OSError('connection lost')
            return super().read(5)
    monkeypatch.setattr(assets, 'urlopen', lambda *a, **kw: Interrupted(data))
    with pytest.raises(OSError):
        assets.download_file('https://example.invalid/model', target, info, lambda *a: None)
    assert not target.exists()
    assert target.with_suffix('.bin.part').read_bytes() == data[:5]
    def resume(request, **kwargs):
        """Honor the caller's byte range with the remaining fixture content."""
        assert request.get_header('Range') == 'bytes=5-'
        response = Response(data[5:]); response.status = 206
        response.headers = {'Content-Range': f'bytes 5-{len(data)-1}/{len(data)}'}
        return response
    monkeypatch.setattr(assets, 'urlopen', resume)
    events = []
    assets.download_file('https://example.invalid/model', target, info, lambda n, total: events.append((n,total)))
    assert target.read_bytes() == data
    assert events[0] == (5,len(data)) and events[-1] == (len(data),len(data))
    assert not target.with_suffix('.bin.part').exists()


def test_server_ignoring_range_restarts_safely(tmp_path, monkeypatch, artifact):
    """A full 200 response must replace partial bytes rather than duplicate them."""
    data, info = artifact
    target = tmp_path / info['name']; target.with_suffix('.bin.part').write_bytes(data[:5])
    monkeypatch.setattr(assets, 'urlopen', lambda *a, **kw: Response(data))
    assets.download_file('https://example.invalid/model', target, info, lambda *a: None)
    assert target.read_bytes() == data


def test_wrong_range_is_rejected_without_publication(tmp_path, monkeypatch, artifact):
    """A mismatched 206 response must never be appended or accepted."""
    data, info = artifact
    target = tmp_path / info['name']; partial = target.with_suffix('.bin.part');partial.write_bytes(data[:5])
    response = Response(data); response.status=206; response.headers={'Content-Range': f'bytes 0-{len(data)-1}/{len(data)}'}
    monkeypatch.setattr(assets, 'urlopen', lambda *a, **kw: response)
    with pytest.raises(RuntimeError, match='range'):
        assets.download_file('https://example.invalid/model', target, info, lambda *a: None)
    assert not target.exists() and partial.read_bytes()==data[:5]


def test_bad_digest_keeps_previous_file_and_allows_clean_retry(tmp_path, monkeypatch, artifact):
    """A same-size tampered download cannot replace the previous cached artifact."""
    data, info = artifact
    target = tmp_path / info['name']; target.write_bytes(b'previous')
    monkeypatch.setattr(assets, 'urlopen', lambda *a, **kw: Response(b'x'*len(data)))
    with pytest.raises(RuntimeError, match='SHA-256'):
        assets.download_file('https://example.invalid/model', target, info, lambda *a: None)
    assert target.read_bytes()==b'previous' and not target.with_suffix('.bin.part').exists()
    monkeypatch.setattr(assets, 'urlopen', lambda *a, **kw: Response(data))
    assets.download_file('https://example.invalid/model', target, info, lambda *a: None)
    assert target.read_bytes()==data


def test_offline_verification_never_connects(tmp_path, monkeypatch, artifact):
    """Require all pinned assets even when the snapshot directory already exists."""
    data, info=artifact
    monkeypatch.setattr(assets, 'MANIFEST', {'files':[info]})
    monkeypatch.setattr(assets, 'florence_snapshot', lambda: tmp_path)
    def forbidden(*a, **kw):
        """Fail any network access during an offline verification."""
        pytest.fail('offline verification opened the network')
    monkeypatch.setattr(assets, 'urlopen', forbidden)
    with pytest.raises(RuntimeError, match='repair'):
        assets.ensure_florence(download=False)
    (tmp_path/info['name']).write_bytes(data)
    assert assets.ensure_florence(download=False)==tmp_path
    (tmp_path/info['name']).write_bytes(b'x'*len(data))
    with pytest.raises(RuntimeError, match='repair'):
        assets.ensure_florence(download=False)


def test_truncated_response_retains_bytes_for_retry(tmp_path, monkeypatch, artifact):
    """A short HTTP body is resumable even when the server closes without an error."""
    data, info = artifact
    target = tmp_path / info['name']
    monkeypatch.setattr(assets, 'urlopen', lambda *a, **kw: Response(data[:5]))
    with pytest.raises(RuntimeError, match='ended early'):
        assets.download_file('https://example.invalid/model', target, info, lambda *a: None)
    assert not target.exists()
    assert target.with_suffix('.bin.part').read_bytes() == data[:5]
