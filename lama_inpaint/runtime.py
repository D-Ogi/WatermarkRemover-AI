"""TorchScript LaMA execution, independent of detection and GUI frameworks."""

import numpy as np
import torch

from .model_store import ModelError, ensure_model, verify_stream
from .processing import InpaintProcessor


def mps_available():
    """Report whether this PyTorch build can place tensors on Apple Metal."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


class LamaInpaint(InpaintProcessor):
    """Load verified LaMA on CPU, CUDA, or Apple MPS and preserve the BGR API."""

    def __init__(self, device="cpu", *, cache_dir=None, download=True):
        """Verify the pinned artifact and load an evaluation-only model."""
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("LaMA supports CPU, CUDA, and MPS")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for LaMA but PyTorch cannot use it")
        if self.device.type == "mps" and not mps_available():
            raise RuntimeError("MPS was requested for LaMA but PyTorch cannot use it")
        path = ensure_model(cache_dir, download=download)
        try:
            # Verify and load through the same handle: never deserialize a replaced path.
            with path.open("rb") as stream:
                verify_stream(stream)
                self.model = (
                    torch.jit.load(stream, map_location="cpu").eval().to(self.device)
                )
        except (OSError, RuntimeError) as exc:
            raise ModelError(
                f"Could not load verified LaMA weights on {self.device}: {exc}"
            ) from exc
        super().__init__(self._predict)

    @torch.inference_mode()
    def _predict(self, image, mask):
        """Run normalized NCHW tensors and validate finite RGB output before quantization."""
        rgb = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        selected = np.ascontiguousarray(mask[None] > 0, dtype=np.int64)
        result = self.model(
            torch.from_numpy(rgb)[None].to(self.device),
            torch.from_numpy(selected)[None].to(self.device),
        )
        if result.shape != (1, 3, *mask.shape) or not torch.isfinite(result).all():
            raise RuntimeError("LaMA returned invalid dimensions or non-finite pixels")
        rgb = result[0].permute(1, 2, 0).cpu().numpy()
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)
