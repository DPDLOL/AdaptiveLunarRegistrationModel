"""Tests for lunar-image preprocessing helpers."""

from __future__ import annotations

import cv2
import numpy as np

from backend.registration.preprocessing import (
    gray8,
    pad_common,
    structural_reps,
)


def test_gray8_converts_color_image_to_uint8_gray() -> None:
    image = np.zeros((64, 48, 3), dtype=np.uint8)
    image[:, :, 0] = 40
    image[:, :, 1] = 80
    image[:, :, 2] = 120

    result = gray8(image)

    assert result.ndim == 2
    assert result.shape == (64, 48)
    assert result.dtype == np.uint8


def test_gray8_preserves_uint8_grayscale_shape() -> None:
    image = np.arange(64 * 48, dtype=np.uint8).reshape(64, 48)

    result = gray8(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_gray8_normalizes_non_uint8_input() -> None:
    image = np.linspace(
        0.0,
        1000.0,
        64 * 48,
        dtype=np.float32,
    ).reshape(64, 48)

    result = gray8(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert int(result.min()) == 0
    assert int(result.max()) == 255


def test_structural_representations_match_image_shape() -> None:
    image = np.zeros((128, 96), dtype=np.uint8)

    cv2.circle(
        image,
        (48, 64),
        20,
        255,
        -1,
    )

    rep_a, rep_b = structural_reps(image)

    assert rep_a.shape == image.shape
    assert rep_b.shape == image.shape
    assert rep_a.dtype == np.uint8
    assert rep_b.dtype == np.uint8


def test_structural_representations_are_not_identical_for_structured_input() -> None:
    image = np.zeros((128, 128), dtype=np.uint8)

    cv2.rectangle(
        image,
        (24, 24),
        (104, 104),
        180,
        -1,
    )

    cv2.circle(
        image,
        (64, 64),
        16,
        255,
        2,
    )

    rep_a, rep_b = structural_reps(image)

    assert np.any(rep_a != rep_b)


def test_pad_common_returns_equal_shapes() -> None:
    image_a = np.ones((80, 100), dtype=np.uint8)
    image_b = np.ones((64, 120), dtype=np.uint8)

    padded_a, padded_b = pad_common(image_a, image_b)

    assert padded_a.shape == padded_b.shape
    assert padded_a.shape == (80, 120)
    assert padded_a.dtype == image_a.dtype
    assert padded_b.dtype == image_b.dtype
