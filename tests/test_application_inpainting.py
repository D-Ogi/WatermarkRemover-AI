"""Exercise image and video application paths without detection/model downloads."""

import cv2
import numpy as np
from PIL import Image
import pytest

pytest.importorskip("torch", reason="requires the application runtime test job")
import remwm
from lama_inpaint.processing import InpaintProcessor


@pytest.fixture
def controlled_inpainting(monkeypatch):
    """Keep detection and prediction deterministic and select the no-audio video path."""

    def mask(image, *args, **kwargs):
        """Select a fixed rectangle in the supplied image coordinates."""
        values = np.zeros((image.height, image.width), np.uint8)
        values[16:32, 16:32] = 255
        return Image.fromarray(values)

    monkeypatch.setattr(remwm, "get_watermark_mask", mask)

    def no_ffmpeg(*args, **kwargs):
        """Choose the no-audio fallback deliberately for the silent fixture."""
        raise FileNotFoundError("controlled no-audio fixture")

    monkeypatch.setattr(remwm.subprocess, "check_output", no_ffmpeg)
    return InpaintProcessor(lambda image, mask: np.full_like(image, (255, 0, 0)))


def test_image_application_path_preserves_source_and_colors(
    tmp_path, controlled_inpainting
):
    """Check the real image routing path, source preservation and RGB/BGR conversion."""
    source, output = tmp_path / "source.png", tmp_path / "output.png"
    Image.new("RGB", (64, 48), (20, 60, 100)).save(source)
    before = source.read_bytes()
    remwm.handle_one(
        source,
        output,
        None,
        None,
        controlled_inpainting,
        "cpu",
        False,
        10,
        "PNG",
        False,
    )
    assert source.read_bytes() == before
    with Image.open(output) as image:
        assert image.size == (64, 48)
        assert image.getpixel((20, 20)) == (255, 0, 0)
        assert image.getpixel((0, 0)) == (20, 60, 100)


@pytest.mark.parametrize("detection_skip", [1, 2])
def test_short_video_paths_keep_frames_and_dimensions(
    tmp_path, controlled_inpainting, monkeypatch, detection_skip
):
    """Exercise both video routes and check frame count, geometry and channel order."""
    source, output = tmp_path / "source.avi", tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    assert writer.isOpened(), "test requires the OpenCV MJPG video writer"
    try:
        for _ in range(3):
            writer.write(np.full((48, 64, 3), (100, 60, 20), np.uint8))
    finally:
        writer.release()
    # Two-pass mode caches detected boxes, while single-pass mode uses the mask.
    monkeypatch.setattr(
        remwm,
        "detect_only",
        lambda *a, **k: [
            {"bbox": [16, 16, 31, 31], "area_percent": 8, "accepted": True}
        ],
    )
    before = source.read_bytes()
    remwm.handle_one(
        source,
        output,
        None,
        None,
        controlled_inpainting,
        "cpu",
        False,
        10,
        "MP4",
        False,
        detection_skip=detection_skip,
    )
    assert source.read_bytes() == before
    capture = cv2.VideoCapture(str(output))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape == (48, 64, 3)
        # Lossy video encoding needs a tolerance; the channel order must survive.
        assert frame[23, 23, 2] > 200
        assert frame[23, 23, 0] < 30
