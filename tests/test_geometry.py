"""Tests for geometric validation helpers."""

from __future__ import annotations

import numpy as np

from backend.registration.geometry import (
    grid_cell_count,
    grid_coverage,
    reproj_stats,
    transform_sanity,
)


def test_identity_homography_is_sane() -> None:
    H = np.eye(3, dtype=np.float64)

    sane, area_ratio, anisotropy = transform_sanity(
        H,
        (256, 256),
    )

    assert sane is True
    assert abs(area_ratio - 1.0) < 1e-6
    assert abs(anisotropy - 1.0) < 1e-6


def test_grid_coverage_uses_multiple_regions() -> None:
    shape = (100, 100)

    points = np.float32(
        [
            [5, 5],
            [95, 5],
            [5, 95],
            [95, 95],
        ]
    )

    coverage = grid_coverage(points, shape)
    cells = grid_cell_count(points, shape)

    assert cells == 4
    assert abs(coverage - 0.25) < 1e-6


def test_empty_points_have_zero_coverage() -> None:
    empty = np.empty((0, 2), dtype=np.float32)

    assert grid_coverage(empty, (100, 100)) == 0.0
    assert grid_cell_count(empty, (100, 100)) == 0


def test_identity_reprojection_has_zero_error() -> None:
    H = np.eye(3, dtype=np.float64)

    src = np.float32(
        [
            [10, 20],
            [50, 40],
            [90, 80],
        ]
    )

    dst = src.copy()

    rms, errors = reproj_stats(H, src, dst)

    assert abs(rms) < 1e-6
    assert errors.shape == (3,)


def test_simple_translation_reprojection() -> None:
    H = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 5.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    src = np.float32(
        [
            [0, 0],
            [20, 10],
            [50, 40],
        ]
    )

    dst = src + np.float32([10, 5])

    rms, errors = reproj_stats(H, src, dst)

    assert abs(rms) < 1e-6
    assert np.all(errors < 1e-6)
