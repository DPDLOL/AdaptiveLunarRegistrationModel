"""Tests for Lucas-Kanade optical-flow refinement helpers."""

from __future__ import annotations

import cv2
import numpy as np

from backend.registration.optical_flow import (
    MAX_DRIFT,
    run_lk,
    select_fast_points,
)


def _textured_image() -> np.ndarray:
    """Create a deterministic image with enough FAST-detectable structure."""
    rng = np.random.default_rng(12345)

    image = rng.integers(
        0,
        256,
        size=(160, 160),
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (20, 20),
        (140, 140),
        255,
        2,
    )

    cv2.line(
        image,
        (20, 20),
        (140, 140),
        0,
        2,
    )

    cv2.line(
        image,
        (140, 20),
        (20, 140),
        0,
        2,
    )

    return image


def test_fast_point_selection_respects_budget() -> None:
    image = _textured_image()

    points = select_fast_points(image, 20)

    assert len(points) <= 20
    assert points.shape[1:] == (1, 2)


def test_lk_identity_homography_tracks_same_image() -> None:
    image = _textured_image()

    H = np.eye(3, dtype=np.float64)

    src, dst = run_lk(
        image,
        image.copy(),
        H,
        budget=100,
    )

    assert len(src) == len(dst)
    assert len(src) >= 10

    drift = np.linalg.norm(
        dst.reshape(-1, 2) - src.reshape(-1, 2),
        axis=1,
    )

    assert np.all(np.isfinite(drift))
    assert float(np.max(drift)) <= MAX_DRIFT


def test_lk_rejects_predictions_outside_target_image() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)

    cv2.rectangle(
        image,
        (20, 20),
        (80, 80),
        255,
        2,
    )

    H = np.array(
        [
            [1.0, 0.0, 1000.0],
            [0.0, 1.0, 1000.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    src, dst = run_lk(
        image,
        image.copy(),
        H,
        budget=100,
    )

    assert len(src) == 0
    assert len(dst) == 0
