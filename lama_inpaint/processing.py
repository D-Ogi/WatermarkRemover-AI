"""LaMA crop/pad/composite processing with an injectable RGB predictor.

Adapted from IOPaint's LaMa/InpaintModel and image helpers (Apache-2.0).
See THIRD_PARTY_NOTICES.md for the source revision and license.
"""

import cv2
import numpy as np


class InpaintProcessor:
    """Process uint8 RGB images and uint8 masks; return independent uint8 BGR.

    The predictor receives padded RGB and single-channel masks and returns RGB.
    A positive mask value selects model input; its 0..255 value controls final
    blending. Large images use the existing 800-pixel crop trigger and 64-pixel
    context margin. Predictions always see the original input, including when
    context crops overlap. Neither caller-owned input is modified.
    """

    def __init__(self, predictor, *, crop_trigger=800, crop_margin=64):
        """Bind an RGB predictor and validate the positive crop trigger and margin."""
        if not isinstance(crop_trigger, int) or crop_trigger < 1:
            raise ValueError("crop_trigger must be a positive integer")
        if not isinstance(crop_margin, int) or crop_margin < 1:
            raise ValueError("crop_margin must be a positive integer")
        self.predictor = predictor
        self.crop_trigger = crop_trigger
        self.crop_margin = crop_margin

    def __call__(self, image, mask):
        """Validate RGB/mask arrays and return a new BGR composite without input mutation."""
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("image must be an HxWx3 uint8 RGB array")
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.uint8
            or mask.shape != image.shape[:2]
        ):
            raise ValueError("mask must be an HxW uint8 array matching the image")
        height, width = mask.shape
        if not height or not width:
            raise ValueError("image and mask must not be empty")
        result = image[:, :, ::-1].copy()
        if not mask.any():
            return result
        if max(height, width) <= self.crop_trigger:
            return self._forward(image, mask)
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            left, right = self._bounds(x, w, width)
            top, bottom = self._bounds(y, h, height)
            crop = self._forward(
                image[top:bottom, left:right], mask[top:bottom, left:right]
            )
            result[top:bottom, left:right] = crop
        return result

    def _bounds(self, start, length, limit):
        # Preserve IOPaint's centered crop sizing, including odd-sized boxes.
        """Return a clipped context interval, preserving legacy odd-box centering."""
        center = (2 * start + length) // 2
        half = (length + 2 * self.crop_margin) // 2
        lower, upper = center - half, center + half
        left, right = max(lower, 0), min(upper, limit)
        if lower < 0:
            right += -lower
        if upper > limit:
            left -= upper - limit
        return max(left, 0), min(right, limit)

    def _forward(self, image, mask):
        """Symmetrically pad one crop, predict RGB, and composite its original-size BGR result."""
        height, width = mask.shape
        # Reflection padding inside the network needs more than a single feature
        # pixel after downsampling. Keep tiny inputs safe without resizing them.
        padded_height = max(32, ((height + 7) // 8) * 8)
        padded_width = max(32, ((width + 7) // 8) * 8)
        padding = ((0, padded_height - height), (0, padded_width - width))
        padded_image = np.pad(image, (*padding, (0, 0)), mode="symmetric")
        padded_mask = np.pad(mask, padding, mode="symmetric")
        prediction = self.predictor(padded_image, padded_mask)
        if (
            not isinstance(prediction, np.ndarray)
            or prediction.shape != padded_image.shape
            or prediction.dtype != np.uint8
        ):
            raise ValueError(
                "LaMA predictor must return a uint8 RGB array matching its padded input"
            )
        prediction = prediction[:height, :width, ::-1]
        # float64 matches the previous IOPaint composite and truncation behavior.
        alpha = mask[:, :, None] / 255.0
        return (prediction * alpha + image[:, :, ::-1] * (1 - alpha)).astype(np.uint8)
