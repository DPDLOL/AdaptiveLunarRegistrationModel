"""Unit tests for the registration validation gates."""

from __future__ import annotations

import math

import numpy as np

from backend.registration.validation import (
    accept_registration,
    final_geometry_ok,
    is_catastrophic_seed,
    seed_is_good,
)


def test_good_seed_passes() -> None:
    H = np.eye(3, dtype=np.float64)

    stats = {
        "seed_inliers": 20,
        "coverage": 0.20,
        "seed_sane": True,
    }

    assert seed_is_good(H, stats) is True


def test_weak_seed_fails() -> None:
    H = np.eye(3, dtype=np.float64)

    stats = {
        "seed_inliers": 19,
        "coverage": 0.20,
        "seed_sane": True,
    }

    assert seed_is_good(H, stats) is False


def test_catastrophic_seed_requires_pathological_geometry() -> None:
    H = np.eye(3, dtype=np.float64)

    assert is_catastrophic_seed(
        H,
        seed_inliers=8,
        seed_area_ratio=0.05,
        seed_anisotropy=1.0,
    ) is True

    assert is_catastrophic_seed(
        H,
        seed_inliers=8,
        seed_area_ratio=1.0,
        seed_anisotropy=1.0,
    ) is False


def test_final_geometry_gate() -> None:
    H = np.eye(3, dtype=np.float64)

    assert final_geometry_ok(
        H,
        coverage=0.50,
        cells=8,
        area_ratio=1.0,
        anisotropy=1.0,
    ) is True

    assert final_geometry_ok(
        H,
        coverage=0.49,
        cells=8,
        area_ratio=1.0,
        anisotropy=1.0,
    ) is False

    assert final_geometry_ok(
        H,
        coverage=0.50,
        cells=8,
        area_ratio=1.0,
        anisotropy=4.01,
    ) is False


def test_final_acceptance_gate() -> None:
    H = np.eye(3, dtype=np.float64)

    assert accept_registration(
        H,
        inliers=100,
        rms=1.5,
        geometry_ok=True,
    ) is True

    assert accept_registration(
        H,
        inliers=99,
        rms=1.0,
        geometry_ok=True,
    ) is False

    assert accept_registration(
        H,
        inliers=100,
        rms=1.5001,
        geometry_ok=True,
    ) is False

    assert accept_registration(
        H,
        inliers=100,
        rms=1.0,
        geometry_ok=False,
    ) is False


def test_invalid_homography_cannot_be_accepted() -> None:
    bad_H = None

    assert final_geometry_ok(
        bad_H,
        coverage=1.0,
        cells=16,
        area_ratio=1.0,
        anisotropy=1.0,
    ) is False

    assert accept_registration(
        bad_H,
        inliers=1000,
        rms=0.1,
        geometry_ok=True,
    ) is False


def test_non_finite_rms_does_not_pass() -> None:
    H = np.eye(3, dtype=np.float64)

    assert accept_registration(
        H,
        inliers=100,
        rms=math.inf,
        geometry_ok=True,
    ) is False
