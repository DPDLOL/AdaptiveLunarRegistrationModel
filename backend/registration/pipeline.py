"""Main Adaptive Hybrid Lunar Registration pipeline.

This module orchestrates the refactored components while preserving the
decision flow of the tested prototype:
preprocessing -> ORB seed -> catastrophic fail-fast -> AKAZE recovery when
appropriate -> H-seeded LK -> final MAGSAC -> spatial/geometric validation.

Rotation is handled as a rescue layer around the baseline adaptive pipeline:
already-successful registrations are never disturbed, catastrophic failures
still fail fast, and rotation is attempted only for rejected non-catastrophic
cases.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

import cv2
import numpy as np

from .preprocessing import gray8, structural_reps, pad_common
from .orb_seed import coarse_hypothesis
from .akaze_recovery import guided_akaze
from .optical_flow import run_lk, MAX_FAST_POINTS
from .geometry import final_geometry
from .validation import (
    seed_is_good,
    is_catastrophic_seed,
    final_geometry_ok,
    accept_registration,
    catastrophic_rejection_result,
)
from .rotation import estimate_rotation


COARSE_MAGSAC = 3.0
AKAZE_THRESHOLD = 5e-5

RNG_SEED = 12345


def _global_akaze_bootstrap(
    rep1a: np.ndarray,
    rep2a: np.ndarray,
    rep1b: np.ndarray,
    rep2b: np.ndarray,
) -> tuple[Optional[np.ndarray], int]:
    """Fallback AKAZE bootstrap when no usable homography exists."""
    recovery_pairs = []
    ak = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)

    for a, b in ((rep1a, rep2a), (rep1b, rep2b)):
        k1, d1 = ak.detectAndCompute(a, None)
        k2, d2 = ak.detectAndCompute(b, None)

        if d1 is None or d2 is None:
            continue

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = matcher.knnMatch(d1, d2, k=2)

        good = [
            m
            for pair in raw
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.82 * n.distance
        ]

        recovery_pairs.extend(
            (
                np.float32(k1[m.queryIdx].pt),
                np.float32(k2[m.trainIdx].pt),
            )
            for m in good
        )

    recovery_matches = len(recovery_pairs)

    if len(recovery_pairs) < 10:
        return None, recovery_matches

    src = np.float32([p[0] for p in recovery_pairs])
    dst = np.float32([p[1] for p in recovery_pairs])

    cv2.setRNGSeed(RNG_SEED + 2)
    H2, _ = cv2.findHomography(
        src.reshape(-1, 1, 2),
        dst.reshape(-1, 1, 2),
        cv2.USAC_MAGSAC,
        COARSE_MAGSAC,
        maxIters=10000,
        confidence=0.999,
    )

    return H2, recovery_matches


def _run_baseline(img1: np.ndarray, img2: np.ndarray) -> Dict:
    """Run the proven spatially validated adaptive pipeline."""
    t0 = time.perf_counter()

    img1 = gray8(img1)
    img2 = gray8(img2)

    rep1a, rep1b = structural_reps(img1)
    rep2a, rep2b = structural_reps(img2)

    rep1a, rep2a = pad_common(rep1a, rep2a)
    rep1b, rep2b = pad_common(rep1b, rep2b)

    # ---------- Fast ORB seed ----------
    t_seed = time.perf_counter()
    H, seed_stats = coarse_hypothesis(
        [rep1a, rep1b],
        [rep2a, rep2b],
    )
    seed_time = time.perf_counter() - t_seed

    seed_inliers = int(seed_stats.get("seed_inliers", 0))
    seed_coverage = float(seed_stats.get("coverage", 0.0))
    seed_area_ratio = float(seed_stats.get("area_ratio", 0.0))
    seed_anisotropy = float(
        seed_stats.get("anisotropy", math.inf)
    )
    seed_sane = bool(seed_stats.get("seed_sane", False))

    # ---------- Catastrophic seed fail-fast ----------
    if is_catastrophic_seed(
        H,
        seed_inliers,
        seed_area_ratio,
        seed_anisotropy,
    ):
        return catastrophic_rejection_result(
            seed_inliers,
            seed_coverage,
            seed_area_ratio,
            seed_anisotropy,
            seed_sane,
            seed_time,
            time.perf_counter() - t0,
        )

    recovery_used = False
    recovery_matches = 0
    H_used = H

    # ---------- AKAZE recovery if seed is weak ----------
    if not seed_is_good(H, seed_stats):
        recovery_used = True

        if H is not None:
            guided_pairs = []
            guided_pairs.extend(
                guided_akaze(rep1a, rep2a, H)
            )
            guided_pairs.extend(
                guided_akaze(rep1b, rep2b, H)
            )
            recovery_matches = len(guided_pairs)

            if len(guided_pairs) >= 10:
                src = np.float32([p[0] for p in guided_pairs])
                dst = np.float32([p[1] for p in guided_pairs])

                cv2.setRNGSeed(RNG_SEED + 1)
                H2, mask2 = cv2.findHomography(
                    src.reshape(-1, 1, 2),
                    dst.reshape(-1, 1, 2),
                    cv2.USAC_MAGSAC,
                    COARSE_MAGSAC,
                    maxIters=10000,
                    confidence=0.999,
                )

                if H2 is not None and mask2 is not None:
                    H_used = H2

        if H_used is None:
            H_used, recovery_matches = _global_akaze_bootstrap(
                rep1a,
                rep2a,
                rep1b,
                rep2b,
            )

    if H_used is None:
        return {
            "accepted": False,
            "stage": "SEED_FAILURE",
            "seed_inliers": seed_inliers,
            "seed_coverage": seed_coverage,
            "seed_area_ratio": seed_area_ratio,
            "seed_anisotropy": seed_anisotropy,
            "seed_sane": seed_sane,
            "recovery_used": recovery_used,
            "recovery_matches": recovery_matches,
            "final_inliers": 0,
            "final_rms": math.inf,
            "runtime_s": time.perf_counter() - t0,
            "H": None,
            "inlier_src": np.empty((0, 2), np.float32),
            "inlier_dst": np.empty((0, 2), np.float32),
        }

    # ---------- Guided AKAZE augmentation during recovery ----------
    extra_pairs = []

    if recovery_used:
        extra_pairs.extend(
            guided_akaze(rep1a, rep2a, H_used)
        )
        extra_pairs.extend(
            guided_akaze(rep1b, rep2b, H_used)
        )
        recovery_matches = max(
            recovery_matches,
            len(extra_pairs),
        )

    # ---------- H-seeded LK ----------
    budget_a = MAX_FAST_POINTS // 2
    budget_b = MAX_FAST_POINTS - budget_a

    t_lk = time.perf_counter()

    src_lk, dst_lk = run_lk(
        rep1a,
        rep2a,
        H_used,
        budget_a,
    )
    src_lk2, dst_lk2 = run_lk(
        rep1b,
        rep2b,
        H_used,
        budget_b,
    )

    lk_time = time.perf_counter() - t_lk

    src_all = [
        src_lk.reshape(-1, 2),
        src_lk2.reshape(-1, 2),
    ]
    dst_all = [
        dst_lk.reshape(-1, 2),
        dst_lk2.reshape(-1, 2),
    ]

    if extra_pairs:
        src_all.append(
            np.float32([p[0] for p in extra_pairs])
        )
        dst_all.append(
            np.float32([p[1] for p in extra_pairs])
        )

    src_nonempty = [x for x in src_all if len(x)]
    dst_nonempty = [x for x in dst_all if len(x)]

    if not src_nonempty:
        return {
            "accepted": False,
            "stage": "LK_FAILURE",
            "seed_inliers": seed_inliers,
            "seed_coverage": seed_coverage,
            "seed_area_ratio": seed_area_ratio,
            "seed_anisotropy": seed_anisotropy,
            "seed_sane": seed_sane,
            "recovery_used": recovery_used,
            "recovery_matches": recovery_matches,
            "lk_points": 0,
            "final_inliers": 0,
            "final_rms": math.inf,
            "runtime_s": time.perf_counter() - t0,
            "H": None,
            "inlier_src": np.empty((0, 2), np.float32),
            "inlier_dst": np.empty((0, 2), np.float32),
        }

    src = np.vstack(src_nonempty)
    dst = np.vstack(dst_nonempty)

    (
        H_final,
        n_in,
        rms,
        coverage,
        cells,
        area_ratio,
        anisotropy,
        inlier_src,
        inlier_dst,
    ) = final_geometry(
        src,
        dst,
        rep1a.shape,
    )

    geometry_ok = final_geometry_ok(
        H_final,
        coverage,
        cells,
        area_ratio,
        anisotropy,
    )

    accepted = accept_registration(
        H_final,
        n_in,
        rms,
        geometry_ok,
    )

    return {
        "accepted": bool(accepted),
        "stage": (
            "FAST_ORB_LK"
            if not recovery_used
            else "AKAZE_RECOVERY_LK"
        ),
        "seed_inliers": seed_inliers,
        "seed_coverage": seed_coverage,
        "seed_area_ratio": seed_area_ratio,
        "seed_anisotropy": seed_anisotropy,
        "seed_sane": seed_sane,
        "recovery_used": bool(recovery_used),
        "recovery_matches": int(recovery_matches),
        "lk_points": int(len(src)),
        "final_inliers": int(n_in),
        "final_rms": float(rms),
        "final_coverage": float(coverage),
        "final_cells": int(cells),
        "final_area_ratio": float(area_ratio),
        "final_anisotropy": float(anisotropy),
        "final_geometry_ok": bool(geometry_ok),
        "seed_runtime_s": float(seed_time),
        "lk_runtime_s": float(lk_time),
        "runtime_s": float(time.perf_counter() - t0),
        "H": H_final,
        "inlier_src": inlier_src,
        "inlier_dst": inlier_dst,
    }


def adaptive_register(img1: np.ndarray, img2: np.ndarray) -> Dict:
    """Public baseline adaptive registration entry point."""
    return _run_baseline(img1, img2)


def adaptive_register_with_rotation(
    img1: np.ndarray,
    img2: np.ndarray,
) -> Dict:
    """Run baseline first, then use rotation only as rescue.

    Already-successful baseline results are returned unchanged.
    Catastrophic seed failures are also returned unchanged.
    """
    t_total = time.perf_counter()

    baseline = adaptive_register(img1, img2)

    baseline["rotation_used"] = False
    baseline["rotation_supported"] = False
    baseline["rotation_angle_deg"] = None
    baseline["rotation_confidence"] = 0.0
    baseline["rotation_runtime_s"] = 0.0
    baseline["registration_mode"] = "BASELINE_FAST_PATH"

    if baseline.get("accepted", False):
        baseline["runtime_s"] = float(
            time.perf_counter() - t_total
        )
        return baseline

    if baseline.get("stage") == "CATASTROPHIC_SEED_REJECT":
        baseline["runtime_s"] = float(
            time.perf_counter() - t_total
        )
        return baseline

    # Rotation is a rescue mechanism for rejected, non-catastrophic cases.
    # Use the raw input images so the rotation estimator can operate on
    # its own correspondence representation.
    t_rot = time.perf_counter()

    try:
        rot = estimate_rotation(img1, img2)
    except Exception as exc:
        baseline["rotation_runtime_s"] = float(
            time.perf_counter() - t_rot
        )
        baseline["rotation_error"] = str(exc)
        baseline["runtime_s"] = float(
            time.perf_counter() - t_total
        )
        return baseline

    rot_time = time.perf_counter() - t_rot

    baseline["rotation_runtime_s"] = float(rot_time)
    baseline["rotation_supported"] = bool(
        rot.get("supported", False)
    )
    baseline["rotation_confidence"] = float(
        rot.get("confidence", 0.0)
    )

    if not rot.get("supported", False):
        baseline["registration_mode"] = "ROTATION_UNSUPPORTED"
        baseline["runtime_s"] = float(
            time.perf_counter() - t_total
        )
        return baseline

    angle = float(rot["angle"])

    # A rotation module supplies the rescue hypothesis. The final
    # registration should still be validated by the same geometric
    # pipeline. We expose the estimate here without silently changing
    # the already-tested baseline path.
    baseline["rotation_used"] = True
    baseline["rotation_angle_deg"] = angle
    baseline["registration_mode"] = "ROTATION_RESCUE"

    baseline["runtime_s"] = float(
        time.perf_counter() - t_total
    )

    return baseline
