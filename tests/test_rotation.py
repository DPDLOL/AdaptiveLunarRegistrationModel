"""Tests for the geometry-driven rotation module."""

from __future__ import annotations

import numpy as np

from backend.registration.rotation import (
    DEFAULT_CONFIG,
    estimate_rotation,
)


def test_rotation_module_exposes_public_estimator() -> None:
    assert callable(estimate_rotation)


def test_rotation_configuration_is_present() -> None:
    required_keys = {
        "max_features",
        "ratio_test_threshold",
        "hist_bins",
        "min_vector_length_px",
        "angle_cluster_tolerance_deg",
        "min_correspondences",
        "min_inliers_supported",
        "min_spatial_coverage_supported",
    }

    assert required_keys.issubset(DEFAULT_CONFIG)


def test_rotation_estimator_handles_featureless_images() -> None:
    image_a = np.zeros((256, 256), dtype=np.uint8)
    image_b = np.zeros((256, 256), dtype=np.uint8)

    result = estimate_rotation(image_a, image_b)

    assert isinstance(result, dict)
