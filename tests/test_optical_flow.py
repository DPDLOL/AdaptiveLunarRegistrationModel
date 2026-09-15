"""Tests for Lucas-Kanade optical-flow refinement helpers."""

from __future__ import annotations

import cv2
import numpy as np

from backend.registration.optical_flow import (
    MAX_DRIFT,
    run_lk,
    select_fast_points,
)


def test_fast_point_selection_respects_budget() -> None:
    image = np.zeros((160, 160), dtype=np.uint8)

    for y in range(20, 140, 20):
        for x in range(20, 140, 20):
            cv2.circle(image, (x, y), 4, 255, -1)

    points = select_fast_points(image, 20)

    assert len(points) <= 20
    assert points.shape[1:] == (1, 2)


def test_lk_identity_homography_tracks_same_image() -> None:
    image = np.zeros((160, 160), dtype=np.uint8)

    for y in range(20, 140, 20):
        for x in range(20, 140, 20):
            cv2.circle(image, (x, y), 4, 255, -1)

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
