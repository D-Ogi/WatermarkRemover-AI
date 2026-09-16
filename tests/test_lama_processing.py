"""Observable crop/pad/composite behavior, with no model download."""

import numpy as np
import pytest

from lama_inpaint.processing import InpaintProcessor


def paint_red(image, mask):
    """Validate padded geometry and return an unmistakable RGB prediction."""
    assert image.shape[:2] == mask.shape
    assert image.shape[0] % 8 == image.shape[1] % 8 == 0
    return np.full_like(image, (255, 0, 0))


@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (16, 16), (801, 19), (17, 803)])
def test_masked_pixels_change_and_originals_survive(shape):
    """Check BGR output, edge masks, exact unmasked pixels and immutable source arrays."""
    image = np.full((*shape, 3), (13, 47, 91), dtype=np.uint8)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[0, 0] = 255
    mask[-1, -1] = 255
    before = image.copy(), mask.copy()
    out = InpaintProcessor(paint_red)(image, mask)
    assert out.shape == image.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(out[mask == 0], image[:, :, ::-1][mask == 0])
    assert np.all(out[mask == 255] == (0, 0, 255))
    np.testing.assert_array_equal(image, before[0])
    np.testing.assert_array_equal(mask, before[1])
    assert not np.shares_memory(out, image)


def test_empty_mask_skips_model_and_returns_independent_bgr():
    """Empty masks must skip prediction and return an independent channel-swapped copy."""
    image = np.full((801, 4, 3), (11, 29, 83), dtype=np.uint8)

    def fail(*args):
        """Fail immediately if an empty-mask request reaches the predictor."""
        pytest.fail("empty mask must not invoke the model")

    out = InpaintProcessor(fail)(image, np.zeros((801, 4), np.uint8))
    np.testing.assert_array_equal(out, image[:, :, ::-1])
    assert not np.shares_memory(out, image)


@pytest.mark.parametrize("shape", [(3, 5), (801, 3)])
def test_full_mask_and_soft_mask(shape):
    """Check full replacement and proportional blending above and below the crop trigger."""
    image = np.full((*shape, 3), (11, 29, 83), dtype=np.uint8)
    mask = np.full(shape, 64, dtype=np.uint8)
    out = InpaintProcessor(paint_red)(image, mask)
    expected = (
        np.array((0, 0, 255)) * (64 / 255) + np.array((83, 29, 11)) * (1 - 64 / 255)
    ).astype(np.uint8)
    assert np.all(out == expected)
    assert np.all(
        InpaintProcessor(paint_red)(image, np.full(shape, 255, np.uint8)) == (0, 0, 255)
    )


def test_overlapping_crops_always_use_original_context():
    """Overlapping context must not feed an earlier prediction into another model call."""
    image = np.full((900, 900, 3), 37, np.uint8)
    mask = np.zeros((900, 900), np.uint8)
    mask[200:205, 200:205] = 255
    mask[220:225, 220:225] = 255
    calls = []

    def predictor(rgb, selected):
        """Inspect the padded source crop and return a controlled prediction."""
        assert np.all(rgb == 37)
        calls.append(rgb.shape)
        return paint_red(rgb, selected)

    out = InpaintProcessor(predictor)(image, mask)
    assert len(calls) == 2
    assert np.all(out[mask == 0] == 37)
    assert np.all(out[mask != 0] == (0, 0, 255))


def test_small_image_symmetric_padding():
    """Extend tiny inputs symmetrically without changing their original-size result."""
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    seen = []

    def predictor(rgb, mask):
        """Inspect the padded source crop and return a controlled prediction."""
        seen.append(rgb.copy())
        return rgb

    out = InpaintProcessor(predictor)(image, np.full((2, 3), 255, np.uint8))
    np.testing.assert_array_equal(out, image[:, :, ::-1])
    assert seen[0].shape == (32, 32, 3)
    np.testing.assert_array_equal(seen[0][0, 3], image[0, 2])
    np.testing.assert_array_equal(seen[0][2, 0], image[1, 0])


@pytest.mark.parametrize(
    "image,mask",
    [
        (np.zeros((0, 2, 3), np.uint8), np.zeros((0, 2), np.uint8)),
        (np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2), np.uint8)),
        (np.zeros((2, 2, 3), np.float32), np.zeros((2, 2), np.uint8)),
        (np.zeros((2, 2, 3), np.uint8), np.zeros((2, 3), np.uint8)),
        (np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), bool)),
    ],
)
def test_reject_invalid_inputs(image, mask):
    """Reject invalid dimensions, channels, dtypes and mismatched masks before inference."""
    with pytest.raises(ValueError):
        InpaintProcessor(paint_red)(image, mask)


def test_reject_invalid_model_output():
    """Reject a predictor result that violates the padded uint8 RGB contract."""
    with pytest.raises(ValueError, match="predictor"):
        InpaintProcessor(lambda *_: np.zeros((8, 8, 3)))(
            np.zeros((2, 2, 3), np.uint8), np.full((2, 2), 255, np.uint8)
        )
