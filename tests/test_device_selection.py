"""Device selection stays explicit for CUDA and Apple MPS."""

import pytest

pytest.importorskip("torch", reason="requires the application runtime test job")
import click

import remwm


def _set_accelerators(monkeypatch, *, cuda, mps):
    monkeypatch.setattr(remwm.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(remwm, "mps_available", lambda: mps)


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    """Auto must not hide an NVIDIA GPU behind Apple MPS or CPU."""
    _set_accelerators(monkeypatch, cuda=True, mps=True)
    assert remwm.resolve_device("auto") == "cuda"
    _set_accelerators(monkeypatch, cuda=False, mps=True)
    assert remwm.resolve_device("auto") == "mps"
    _set_accelerators(monkeypatch, cuda=False, mps=False)
    assert remwm.resolve_device("auto") == "cpu"


def test_named_device_does_not_fall_back(monkeypatch):
    """A requested accelerator that is missing fails instead of switching devices."""
    _set_accelerators(monkeypatch, cuda=False, mps=False)
    with pytest.raises(click.ClickException, match="CUDA"):
        remwm.resolve_device("cuda")
    with pytest.raises(click.ClickException, match="MPS"):
        remwm.resolve_device("mps")
    assert remwm.resolve_device("cpu") == "cpu"
    _set_accelerators(monkeypatch, cuda=False, mps=True)
    assert remwm.resolve_device("MPS") == "mps"


def test_mps_uses_float32():
    """Florence generation on MPS stays in float32."""
    assert remwm.inference_dtype("mps") is remwm.torch.float32
    assert remwm.inference_dtype("cpu") is remwm.torch.float32
    assert remwm.inference_dtype("cuda") is None
