"""Basic import and API tests for SIH26166."""

from __future__ import annotations

import cv2
import numpy as np


def test_opencv_available() -> None:
    """OpenCV must be importable."""
    assert cv2.__version__


def test_akaze_available() -> None:
    """The recovery path requires OpenCV's AKAZE implementation."""
    assert hasattr(cv2, "AKAZE_create")
    detector = cv2.AKAZE_create()
    assert detector is not None


def test_registration_pipeline_public_api() -> None:
    """The package must expose its public registration entry points."""
    from backend.registration.pipeline import (
        adaptive_register,
        adaptive_register_with_rotation,
    )

    assert callable(adaptive_register)
    assert callable(adaptive_register_with_rotation)


def test_visualization_public_api() -> None:
    """The visualization helper must be available to the backend."""
    from backend.registration.visualization import (
        save_inlier_visualization,
    )

    assert callable(save_inlier_visualization)


def test_basic_numpy_array() -> None:
    """Sanity check for the numerical dependency used by registration."""
    image = np.zeros((32, 32), dtype=np.uint8)
    assert image.shape == (32, 32)
    assert image.dtype == np.uint8
