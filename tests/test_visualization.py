"""Tests for inlier visualization output."""

from __future__ import annotations

import cv2
import numpy as np

from backend.registration.visualization import save_inlier_visualization


def test_inlier_visualization_is_written(tmp_path) -> None:
    image_a = np.zeros((120, 160), dtype=np.uint8)
    image_b = np.zeros((120, 160), dtype=np.uint8)

    cv2.circle(image_a, (40, 40), 10, 255, -1)
    cv2.circle(image_b, (45, 45), 10, 255, -1)

    src = np.float32(
        [
            [40, 40],
            [60, 50],
            [80, 70],
        ]
    )

    dst = src + np.float32([5, 5])

    output = tmp_path / "inliers.png"

    save_inlier_visualization(
        image_a,
        image_b,
        src,
        dst,
        output,
        "test pair",
        True,
    )

    assert output.is_file()

    rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)

    assert rendered is not None
    assert rendered.ndim == 3
    assert rendered.shape[0] >= 120
    assert rendered.shape[1] >= 320


def test_inlier_visualization_handles_empty_matches(tmp_path) -> None:
    image_a = np.zeros((80, 100), dtype=np.uint8)
    image_b = np.zeros((80, 100), dtype=np.uint8)

    empty = np.empty((0, 2), dtype=np.float32)

    output = tmp_path / "empty_inliers.png"

    save_inlier_visualization(
        image_a,
        image_b,
        empty,
        empty,
        output,
        "empty pair",
        False,
    )

    assert output.is_file()
    assert cv2.imread(str(output), cv2.IMREAD_COLOR) is not None
