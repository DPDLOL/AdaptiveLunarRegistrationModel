"""Adaptive decision and validation logic for lunar registration.

This module contains the prototype's seed-quality checks, catastrophic-seed
fail-fast gate, and final acceptance criteria. Thresholds are preserved.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


MIN_SEED_INLIERS = 20
MIN_SEED_COVERAGE = 0.20

CATASTROPHIC_MAX_INLIERS = 8
CATASTROPHIC_MIN_AREA_RATIO = 0.10
CATASTROPHIC_MAX_AREA_RATIO = 10.0
CATASTROPHIC_MAX_ANISOTROPY = 6.0

MIN_FINAL_INLIERS = 100
MAX_FINAL_RMS = 1.5

MIN_FINAL_COVERAGE = 0.50
MIN_FINAL_CELLS = 8
MIN_FINAL_AREA_RATIO = 0.20
MAX_FINAL_AREA_RATIO = 5.0
MAX_FINAL_ANISOTROPY = 4.0


def seed_is_good(H: Optional[np.ndarray], stats: Dict) -> bool:
    """Return True when the coarse seed is usable by the fast LK path."""
    return bool(
        H is not None
        and stats.get("seed_inliers", 0) >= MIN_SEED_INLIERS
        and stats.get("coverage", 0.0) >= MIN_SEED_COVERAGE
        and stats.get("seed_sane", False)
    )


def is_catastrophic_seed(
    H: Optional[np.ndarray],
    seed_inliers: int,
    seed_area_ratio: float,
    seed_anisotropy: float,
) -> bool:
    """Reject only very small seeds with clearly pathological geometry."""
    return bool(
        H is not None
        and seed_inliers <= CATASTROPHIC_MAX_INLIERS
        and (
            seed_area_ratio < CATASTROPHIC_MIN_AREA_RATIO
            or seed_area_ratio > CATASTROPHIC_MAX_AREA_RATIO
            or seed_anisotropy > CATASTROPHIC_MAX_ANISOTROPY
        )
    )


def final_geometry_ok(
    H: Optional[np.ndarray],
    coverage: float,
    cells: int,
    area_ratio: float,
    anisotropy: float,
) -> bool:
    """Check the final global-geometry acceptance gate."""
    return bool(
        H is not None
        and coverage >= MIN_FINAL_COVERAGE
        and cells >= MIN_FINAL_CELLS
        and MIN_FINAL_AREA_RATIO <= area_ratio <= MAX_FINAL_AREA_RATIO
        and anisotropy <= MAX_FINAL_ANISOTROPY
    )


def accept_registration(
    H: Optional[np.ndarray],
    inliers: int,
    rms: float,
    geometry_ok: bool,
) -> bool:
    """Apply the final acceptance criteria."""
    return bool(
        H is not None
        and inliers >= MIN_FINAL_INLIERS
        and rms <= MAX_FINAL_RMS
        and geometry_ok
    )


def catastrophic_rejection_result(
    seed_inliers: int,
    seed_coverage: float,
    seed_area_ratio: float,
    seed_anisotropy: float,
    seed_sane: bool,
    seed_runtime_s: float,
    runtime_s: float,
) -> Dict:
    """Build the same fail-fast result structure used by the prototype."""
    return {
        "accepted": False,
        "stage": "CATASTROPHIC_SEED_REJECT",
        "seed_inliers": int(seed_inliers),
        "seed_coverage": float(seed_coverage),
        "seed_area_ratio": float(seed_area_ratio),
        "seed_anisotropy": float(seed_anisotropy),
        "seed_sane": bool(seed_sane),
        "recovery_used": False,
        "recovery_matches": 0,
        "lk_points": 0,
        "final_inliers": 0,
        "final_rms": math.inf,
        "final_coverage": 0.0,
        "final_cells": 0,
        "final_area_ratio": 0.0,
        "final_anisotropy": math.inf,
        "final_geometry_ok": False,
        "seed_runtime_s": float(seed_runtime_s),
        "lk_runtime_s": 0.0,
        "runtime_s": float(runtime_s),
        "H": None,
        "inlier_src": np.empty((0, 2), np.float32),
        "inlier_dst": np.empty((0, 2), np.float32),
    }
