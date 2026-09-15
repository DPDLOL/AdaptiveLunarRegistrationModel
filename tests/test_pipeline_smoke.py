"""Pipeline-level smoke tests for the SIH26166 registration system."""

from __future__ import annotations

import cv2
import numpy as np

from backend.registration.pipeline import adaptive_register_with_rotation


def test_blank_images_are_rejected_without_exception() -> None:
    """Featureless inputs should fail safely instead of crashing."""
    image_a = np.zeros((256, 256), dtype=np.uint8)
    image_b = np.zeros((256, 256), dtype=np.uint8)

    result = adaptive_register_with_rotation(image_a, image_b)

    assert isinstance(result, dict)
    assert result.get("accepted") is False


def test_color_inputs_are_supported() -> None:
    """The public API should accept ordinary BGR images."""
    image_a = np.zeros((128, 128, 3), dtype=np.uint8)
    image_b = np.zeros((128, 128, 3), dtype=np.uint8)

    cv2.circle(image_a, (64, 64), 20, (255, 255, 255), -1)
    cv2.circle(image_b, (64, 64), 20, (255, 255, 255), -1)

    result = adaptive_register_with_rotation(image_a, image_b)

    assert isinstance(result, dict)
    assert "accepted" in result
    assert "stage" in result
