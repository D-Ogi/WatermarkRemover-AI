"""Real TorchScript execution on a generated fixture; no downloaded models."""

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the application runtime test job")
from lama_inpaint import model_store
from lama_inpaint.runtime import LamaInpaint


@pytest.fixture
def scripted_weights(tmp_path, monkeypatch):
    """Create and checksum a tiny local TorchScript fixture without model downloads."""

    class Paint(torch.nn.Module):
        def forward(self, image, mask):
            # Exercise both tensor inputs; model writes red into selected pixels.
            """Use both tensor inputs to replace selected pixels with RGB red."""
            color = torch.zeros_like(image)
            color[:, 0] = 1
            return image * (1 - mask) + color * mask

    traced = torch.jit.trace(
        Paint(), (torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 8, 8, dtype=torch.int64))
    )
    path = tmp_path / model_store.MODEL_NAME
    torch.jit.save(traced, path)
    data = path.read_bytes()
    monkeypatch.setattr(model_store, "MODEL_SIZE", len(data))
    monkeypatch.setattr(model_store, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    return tmp_path


def test_cpu_scripted_model_end_to_end(scripted_weights):
    """Exercise verified loading, tensor conversion and compositing with a real TorchScript call."""
    model = LamaInpaint("cpu", cache_dir=scripted_weights, download=False)
    image = np.full((13, 17, 3), (23, 51, 81), np.uint8)
    mask = np.zeros((13, 17), np.uint8)
    mask[2:7, 3:9] = 255
    result = model(image, mask)
    assert np.all(result[mask == 255] == (0, 0, 255))
    np.testing.assert_array_equal(result[mask == 0], image[:, :, ::-1][mask == 0])


def test_corrupt_weights_are_never_deserialized(scripted_weights, monkeypatch):
    """Reject modified artifact bytes before invoking the TorchScript loader."""
    (scripted_weights / model_store.MODEL_NAME).write_bytes(b"corrupt")
    monkeypatch.setattr(
        torch.jit,
        "load",
        lambda *a, **k: pytest.fail("must not load unverified weights"),
    )
    with pytest.raises(model_store.ModelError):
        LamaInpaint("cpu", cache_dir=scripted_weights, download=False)


def test_reject_unvalidated_device_before_loading():
    """Reject MPS explicitly rather than promising unvalidated operator support."""
    with pytest.raises(ValueError, match="MPS"):
        LamaInpaint("mps", download=False)


def test_cuda_unavailable_is_explicit(monkeypatch):
    """Report unavailable CUDA without silently selecting a different device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        LamaInpaint("cuda", download=False)
