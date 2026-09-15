"""Rotation estimation module for Adaptive Hybrid Lunar Registration.

This module contains the tested geometry-driven rotation estimator and its
helper functions. The numerical logic is preserved from the working
prototype; this file is a refactor for repository organization.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

DEFAULT_CONFIG: Dict[str, any] = {
    "max_features": 3000,
    "ratio_test_threshold": 0.75,
    "matrix_ransac_threshold_px": 4.0,
    "matrix_ransac_max_iters": 2000,
    "matrix_ransac_confidence": 0.995,
    "matrix_ransac_refine_iters": 10,


    "hist_bins": 36,
    "min_vector_length_px": 3.0,
    "max_pairwise_samples": 6000,
    "max_peaks_per_source": 2,
    "min_peak_vote_frac": 0.04,
    "angle_cluster_tolerance_deg": 8.0,


    "geometric_inlier_threshold_frac": 0.015,
    "geometric_refine_iterations": 2,


    "refine_search_range_deg": 5.0,
    "refine_step_deg": 0.25,


    "refine_large_delta_threshold_deg": 2.0,


    "min_correspondences": 4,
    "min_inliers_supported": 6,
    "min_inlier_ratio_supported": 0.25,
    "min_spatial_coverage_supported": 0.30,
    "min_spatial_cells_supported": 3,
    "spatial_grid_size": 4,
    "spatial_coverage_power": 0.5,
    "single_source_ratio_penalty": 0.85,

    "ambiguity_score_ratio": 0.6,

    "ambiguity_reject_ratio": 0.50,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def estimate_rotation(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: Optional[np.ndarray] = None,
    points_b: Optional[np.ndarray] = None,
    angles_a: Optional[np.ndarray] = None,
    angles_b: Optional[np.ndarray] = None,
    config: Optional[Dict[str, any]] = None,
) -> Dict[str, any]:

    t0 = time.time()
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    diagonal = _image_diagonal(image_a, image_b)

    # ------------------------------------------------------------------
    # 1. Obtain correspondences (use provided ones; else fall back to an
    #    internal SIFT detector+matcher for standalone use/testing).
    # ------------------------------------------------------------------
    src_note = "provided"
    if points_a is None or points_b is None:
        points_a, points_b, angles_a, angles_b = _detect_and_match(
            image_a, image_b, cfg
        )
        src_note = "internal_sift_fallback"
    else:
        points_a = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
        points_b = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
        if angles_a is not None:
            angles_a = np.asarray(angles_a, dtype=np.float64).reshape(-1)
        if angles_b is not None:
            angles_b = np.asarray(angles_b, dtype=np.float64).reshape(-1)

    n = len(points_a)
    if n < cfg["min_correspondences"]:
        return _empty_result(
            reason=f"only {n} correspondence(s) available "
                   f"(need >= {cfg['min_correspondences']})",
            runtime=time.time() - t0,
            num_correspondences=n,
            correspondence_source=src_note,
        )

    # ------------------------------------------------------------------
    # 2. Generate a SMALL set of angle hypotheses from independent
    #    evidence sources.
    # ------------------------------------------------------------------
    raw_hypotheses: List[Dict[str, any]] = []

    have_orientations = (
        angles_a is not None and angles_b is not None
        and len(angles_a) == n and len(angles_b) == n
    )
    if have_orientations:
        raw_hypotheses += _orientation_histogram_hypotheses(
            angles_a, angles_b, cfg
        )

    raw_hypotheses += _pairwise_vector_hypotheses(points_a, points_b, cfg)
    raw_hypotheses += _similarity_matrix_hypothesis(points_a, points_b, cfg)


    raw_hypotheses.append(
        {"angle": 0.0, "source": "baseline", "votes": 0, "vote_frac": 0.0}
    )

    if not raw_hypotheses:
        return _empty_result(
            reason="no angle evidence could be extracted from correspondences",
            runtime=time.time() - t0,
            num_correspondences=n,
            correspondence_source=src_note,
        )

    # ------------------------------------------------------------------
    # 3. Cluster hypotheses from different sources that agree, so we can
    #    tell "two independent methods agree" apart from "one method
    #    produced two nearby bins".
    # ------------------------------------------------------------------
    clusters = _cluster_hypotheses(raw_hypotheses, cfg)

    # ------------------------------------------------------------------
    # 4. For every cluster, also test its 180-degree-flipped variant.
    #    This is the explicit fix for the recurring 180-degree-ambiguity
    #    failure mode -- we never assume the sign/direction is right,
    #    we let geometry decide.
    # ------------------------------------------------------------------
    candidates = []
    for c in clusters:
        candidates.append(dict(c, is_flip_variant=False, parent_angle=None))
        flip_angle = _wrap_angle(c["angle"] + 180.0)
        candidates.append({
            "angle": flip_angle,
            "sources": c["sources"],
            "votes": c["votes"],
            "vote_frac": c["vote_frac"],
            "is_flip_variant": True,
            "parent_angle": c["angle"],
        })

    # ------------------------------------------------------------------
    # 5. Lightweight geometric validation of every candidate.
    # ------------------------------------------------------------------
    inlier_threshold = cfg["geometric_inlier_threshold_frac"] * diagonal
    for cand in candidates:
        geo = _geometric_validate(
            points_a, points_b, cand["angle"], inlier_threshold,
            cfg["geometric_refine_iterations"],
        )
        cand.update(geo)
        cand.update(_spatial_support(
            points_a, points_b, cand["angle"], inlier_threshold,
            cfg["spatial_grid_size"], image_a.shape, image_b.shape
        ))


    candidates = _dedupe_candidates(candidates, cfg)

    # ------------------------------------------------------------------
    # 6. Rank and select.
    # ------------------------------------------------------------------
    candidates.sort(
        key=lambda c: (
            c["inlier_ratio"] * (0.85 + 0.15 * c.get("spatial_coverage", 0.0)),
            c["vote_frac"], -c["avg_residual"]
        ),
        reverse=True,
    )
    best = candidates[0]

    # ------------------------------------------------------------------
    # 7. Local angle refinement.
    #    The initial angle comes from a coarse histogram peak. A local
    #    grid search finds the angle that truly maximises inliers and
    #    minimises residual within the tight geometric gate. This fixes
    #    the "off by 3 degrees on repetitive terrain" failure mode.
    # ------------------------------------------------------------------
    refined_angle, refined_geo = _refine_angle(
        points_a, points_b,
        best["angle"],
        inlier_threshold,
        cfg["refine_search_range_deg"],
        cfg["refine_step_deg"],
        cfg["geometric_refine_iterations"],
    )

    # Update best with refined values
    best["angle"] = refined_angle
    best["inliers"] = refined_geo["inliers"]
    best["inlier_ratio"] = refined_geo["inlier_ratio"]
    best["avg_residual"] = refined_geo["avg_residual"]
    best["refined"] = True
    best["refinement_delta_deg"] = _angular_distance(refined_angle, candidates[0]["angle"])

    # Re-sort candidates with refined best at top (it should still win,
    # but in pathological cases refinement could make it worse).
    candidates[0] = best
    candidates.sort(
        key=lambda c: (c["inlier_ratio"], c.get("vote_frac", 0.0), -c.get("avg_residual", 0.0)),
        reverse=True,
    )
    best = candidates[0]

    # ------------------------------------------------------------------
    # 8. 180-degree ambiguity analysis.
    #    We compute BOTH the ambiguity flag (for diagnostics) and the
    #    rejection criterion (for correctness).
    # ------------------------------------------------------------------
    antipode = _wrap_angle(best["angle"] + 180.0)
    ambiguity_180 = False
    flip_inlier_ratio = 0.0
    for cand in candidates[1:]:
        if _angular_distance(cand["angle"], antipode) <= cfg["angle_cluster_tolerance_deg"]:
            flip_inlier_ratio = cand["inlier_ratio"]
            if best["inlier_ratio"] > 0 and \
               cand["inlier_ratio"] >= cfg["ambiguity_score_ratio"] * best["inlier_ratio"]:
                ambiguity_180 = True
            break


    severely_ambiguous = (
        best["inlier_ratio"] > 0
        and flip_inlier_ratio >= cfg["ambiguity_reject_ratio"] * best["inlier_ratio"]
    )

    num_independent_sources = len(set(best.get("sources", [])) - {"baseline"})
    spatial_ok = (
        best.get("spatial_coverage", 0.0) >= cfg["min_spatial_coverage_supported"]
        and best.get("spatial_cells", 0) >= cfg["min_spatial_cells_supported"]
    )
    single_source_weak = (
        num_independent_sources <= 1 and best["inlier_ratio"] < 0.40
    )
    supported = (
        best["inliers"] >= cfg["min_inliers_supported"]
        and best["inlier_ratio"] >= cfg["min_inlier_ratio_supported"]
        and not severely_ambiguous
        and (spatial_ok or best["inlier_ratio"] >= 0.50)
        and not single_source_weak
    )

    # ------------------------------------------------------------------
    # 9. Confidence: combine geometric consistency, source agreement,
    #    vote strength, and refinement quality; penalize unresolved
    #    180-degree ambiguity.
    # ------------------------------------------------------------------
    agreement_score = min(num_independent_sources / 2.0, 1.0)

    confidence = (
        0.50 * best["inlier_ratio"]
        + 0.25 * agreement_score
        + 0.15 * min(best.get("vote_frac", 0.0) * 3.0, 1.0)
        + 0.10 * (1.0 if best.get("refinement_delta_deg", 0.0) < cfg["refine_large_delta_threshold_deg"] else 0.5)
    )

    confidence *= (0.8 + 0.2 * best.get("spatial_coverage", 0.0))
    if num_independent_sources <= 1:
        confidence *= cfg["single_source_ratio_penalty"]
    if ambiguity_180:
        confidence *= 0.7
    if severely_ambiguous:
        confidence = min(confidence, 0.30)
    if not supported:
        confidence = min(confidence, 0.35)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if supported:
        reason = (
            f"angle {best['angle']:.1f} deg supported by "
            f"{best['inliers']}/{n} geometric inliers "
            f"({best['inlier_ratio']*100:.0f}%) "
            f"from {max(num_independent_sources, 1)} evidence source(s)"
        )
        if best.get("refinement_delta_deg", 0.0) >= cfg["refine_large_delta_threshold_deg"]:
            reason += f" (refined by {best['refinement_delta_deg']:.1f} deg from coarse estimate)"
    elif severely_ambiguous:
        reason = (
            f"best candidate ({best['angle']:.1f} deg) is severely ambiguous: "
            f"flip variant ({antipode:.1f} deg) scored {flip_inlier_ratio*100:.0f}% "
            f"vs winner {best['inlier_ratio']*100:.0f}% -- cannot reliably distinguish "
            f"theta from theta+180"
        )
    else:
        reason = (
            f"best candidate ({best['angle']:.1f} deg) only reached "
            f"{best['inliers']}/{n} inliers "
            f"({best['inlier_ratio']*100:.0f}%) -- below support threshold"
        )

    result = {
        "angle": float(best["angle"]) if supported else None,
        "confidence": confidence,
        "supported": supported,
        "hypotheses": _round_candidates_for_output(candidates),
        "selected": _round_candidate(best),
        "ambiguity_180": ambiguity_180,
        "severely_ambiguous": severely_ambiguous,
        "flip_inlier_ratio": float(flip_inlier_ratio),
        "num_correspondences": n,
        "num_inliers": int(best["inliers"]),
        "inlier_ratio": float(best["inlier_ratio"]),
        "spatial_coverage": float(best.get("spatial_coverage", 0.0)),
        "spatial_cells": int(best.get("spatial_cells", 0)),
        "sources_used": sorted(set(best.get("sources", []))),
        "correspondence_source": src_note,
        "reason": reason,
        "runtime_seconds": time.time() - t0,
    }
    return result


# ---------------------------------------------------------------------------
# Internal fallback feature detection + matching
# ---------------------------------------------------------------------------

def _detect_and_match(image_a, image_b, cfg):
    """Only used when the caller doesn't supply correspondences already
    (e.g. standalone testing of this module). The real pipeline should
    pass in points_a/points_b from its RootSIFT/LoFTR stage instead."""
    gray_a = _to_gray(image_a)
    gray_b = _to_gray(image_b)

    sift = cv2.SIFT_create(nfeatures=cfg["max_features"])
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return (np.zeros((0, 2)), np.zeros((0, 2)),
                np.zeros((0,)), np.zeros((0,)))

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des_a, des_b, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, nn = pair
        if m.distance < cfg["ratio_test_threshold"] * nn.distance:
            good.append(m)

    points_a = np.array([kp_a[m.queryIdx].pt for m in good], dtype=np.float64)
    points_b = np.array([kp_b[m.trainIdx].pt for m in good], dtype=np.float64)
    angles_a = np.array([kp_a[m.queryIdx].angle for m in good], dtype=np.float64)
    angles_b = np.array([kp_b[m.trainIdx].angle for m in good], dtype=np.float64)

    return points_a, points_b, angles_a, angles_b


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _image_diagonal(image_a, image_b) -> float:
    h = max(image_a.shape[0], image_b.shape[0])
    w = max(image_a.shape[1], image_b.shape[1])
    return float(math.hypot(h, w))


# ---------------------------------------------------------------------------
# Hypothesis generation
# ---------------------------------------------------------------------------

def _wrap_angle(deg: float) -> float:
    """Wrap an angle into [0, 360)."""
    return deg % 360.0


def _angular_distance(a: float, b: float) -> float:
    """Smallest difference between two angles in degrees, in [0, 180]."""
    d = abs(_wrap_angle(a) - _wrap_angle(b))
    return min(d, 360.0 - d)


def _circular_hist_peaks(angles: np.ndarray, weights: np.ndarray, cfg,
                          source_name: str) -> List[Dict[str, any]]:
    """Build a weighted circular histogram over [0, 360) and return the
    top few peaks as hypotheses, using circular-mean refinement so a
    peak straddling the 0/360 wrap point is not split in two."""
    n_bins = cfg["hist_bins"]
    bin_width = 360.0 / n_bins

    if len(angles) == 0:
        return []

    total_weight = weights.sum()
    if total_weight <= 0:
        return []

    bin_idx = (angles // bin_width).astype(int) % n_bins
    hist = np.zeros(n_bins, dtype=np.float64)
    for idx, w in zip(bin_idx, weights):
        hist[idx] += w

    peaks = []
    hist_work = hist.copy()
    for _ in range(cfg["max_peaks_per_source"]):
        top_bin = int(np.argmax(hist_work))
        if hist_work[top_bin] <= 0:
            break
        vote_frac = hist[top_bin] / total_weight
        if vote_frac < cfg["min_peak_vote_frac"]:
            break


        mask = np.array([_angular_distance(a, top_bin * bin_width + bin_width / 2)
                          <= bin_width for a in angles])
        if mask.any():
            refined_angle = _weighted_circular_mean(angles[mask], weights[mask])
        else:
            refined_angle = top_bin * bin_width + bin_width / 2

        peaks.append({
            "angle": _wrap_angle(refined_angle),
            "source": source_name,
            "votes": float(hist[top_bin]),
            "vote_frac": float(vote_frac),
        })


        suppress_bins = [top_bin]
        suppress_bins.append((top_bin - 1) % n_bins)
        suppress_bins.append((top_bin + 1) % n_bins)
        for b in suppress_bins:
            hist_work[b] = 0.0

    return peaks


def _weighted_circular_mean(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    rad = np.radians(angles_deg)
    x = np.sum(weights * np.cos(rad))
    y = np.sum(weights * np.sin(rad))
    if x == 0 and y == 0:
        return float(np.mean(angles_deg))
    return math.degrees(math.atan2(y, x)) % 360.0


def _orientation_histogram_hypotheses(angles_a, angles_b, cfg) -> List[Dict[str, any]]:

    diffs = np.array([_wrap_angle(b - a) for a, b in zip(angles_a, angles_b)])
    weights = np.ones_like(diffs)

    peaks = _circular_hist_peaks(diffs, weights, cfg, "orientation_hist")

    hedged = []
    for p in peaks:
        hedged.append(p)
        negated = dict(p)
        negated["angle"] = _wrap_angle(-p["angle"])
        negated["source"] = "orientation_hist_sign_hedge"
        hedged.append(negated)
    return hedged


def _pairwise_vector_hypotheses(points_a, points_b, cfg) -> List[Dict[str, any]]:

    n = len(points_a)
    if n < 2:
        return []

    max_pairs = min(cfg["max_pairwise_samples"], n * (n - 1) // 2)
    rng = np.random.default_rng(12345)

    if n <= 140:  # small enough to just use all pairs
        idx_i, idx_j = np.triu_indices(n, k=1)
    else:
        idx_i = rng.integers(0, n, size=max_pairs * 2)
        idx_j = rng.integers(0, n, size=max_pairs * 2)
        keep = idx_i != idx_j
        idx_i, idx_j = idx_i[keep], idx_j[keep]
        idx_i, idx_j = idx_i[:max_pairs], idx_j[:max_pairs]

    va = points_a[idx_j] - points_a[idx_i]
    vb = points_b[idx_j] - points_b[idx_i]

    len_a = np.linalg.norm(va, axis=1)
    len_b = np.linalg.norm(vb, axis=1)
    valid = (len_a >= cfg["min_vector_length_px"]) & (len_b >= cfg["min_vector_length_px"])
    va, vb, len_a, len_b = va[valid], vb[valid], len_a[valid], len_b[valid]

    if len(va) == 0:
        return []

    cross = va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0]
    dot = va[:, 0] * vb[:, 0] + va[:, 1] * vb[:, 1]

    angles = (-np.degrees(np.arctan2(cross, dot))) % 360.0

    weights = np.minimum(len_a, len_b)

    return _circular_hist_peaks(angles, weights, cfg, "pairwise_vector")



# ---------------------------------------------------------------------------
# Robust 2D similarity-transform hypothesis (rotation-matrix evidence)
# ---------------------------------------------------------------------------

def _similarity_matrix_hypothesis(points_a, points_b, cfg) -> List[Dict[str, any]]:

    if len(points_a) < 4:
        return []

    M, inlier_mask = cv2.estimateAffinePartial2D(
        np.asarray(points_a, dtype=np.float32),
        np.asarray(points_b, dtype=np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg["matrix_ransac_threshold_px"]),
        maxIters=int(cfg["matrix_ransac_max_iters"]),
        confidence=float(cfg["matrix_ransac_confidence"]),
        refineIters=int(cfg["matrix_ransac_refine_iters"]),
    )
    if M is None:
        return []

    a, b = float(M[0, 0]), float(M[0, 1])
    c, d = float(M[1, 0]), float(M[1, 1])
    scale = math.sqrt(max((a*a + b*b + c*c + d*d) / 2.0, 0.0))
    if scale <= 1e-8:
        return []

    # Screen-coordinate convention used by _rotation_matrix:
    # [[cos(t), sin(t)], [-sin(t), cos(t)]].
    angle = math.degrees(math.atan2(b, a)) % 360.0

    matrix_inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
    matrix_ratio = matrix_inliers / max(len(points_a), 1)

    # Reject obviously non-similarity fits (scale should be plausible).
    return [{
        "angle": angle,
        "source": "similarity_matrix",
        "votes": float(matrix_inliers),
        "vote_frac": float(matrix_ratio),
        "matrix_scale": float(scale),
        "matrix_inliers": matrix_inliers,
        "matrix_inlier_ratio": float(matrix_ratio),
    }]


# ---------------------------------------------------------------------------
# Clustering hypotheses across sources
# ---------------------------------------------------------------------------

def _cluster_hypotheses(raw: List[Dict[str, any]], cfg) -> List[Dict[str, any]]:

    tol = cfg["angle_cluster_tolerance_deg"]
    remaining = sorted(raw, key=lambda h: h.get("votes", 0.0), reverse=True)
    clusters: List[Dict[str, any]] = []

    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        rest = []
        for h in remaining:
            if _angular_distance(h["angle"], seed["angle"]) <= tol:
                members.append(h)
            else:
                rest.append(h)
        remaining = rest

        angles = np.array([m["angle"] for m in members])
        weights = np.array([max(m.get("votes", 0.0), 1e-6) for m in members])
        merged_angle = _weighted_circular_mean(angles, weights)

        clusters.append({
            "angle": merged_angle,
            "sources": [m["source"] for m in members],
            "votes": float(sum(m.get("votes", 0.0) for m in members)),
            "vote_frac": float(sum(m.get("vote_frac", 0.0) for m in members)),
        })

    return clusters


# ---------------------------------------------------------------------------
# Geometric validation
# ---------------------------------------------------------------------------

def _rotation_matrix(angle_deg: float) -> np.ndarray:

    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, s], [-s, c]])


def _geometric_validate(points_a, points_b, angle_deg, inlier_threshold,
                         refine_iterations) -> Dict[str, any]:

    R = _rotation_matrix(angle_deg)
    rotated_a = points_a @ R.T

    translation = np.median(points_b - rotated_a, axis=0)
    mask = np.ones(len(points_a), dtype=bool)

    for _ in range(max(refine_iterations, 1)):
        predicted_b = rotated_a + translation
        residuals = np.linalg.norm(predicted_b - points_b, axis=1)
        mask = residuals < inlier_threshold
        if mask.sum() == 0:
            break
        translation = np.median(points_b[mask] - rotated_a[mask], axis=0)

    predicted_b = rotated_a + translation
    residuals = np.linalg.norm(predicted_b - points_b, axis=1)
    mask = residuals < inlier_threshold

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(points_a)
    avg_residual = float(residuals[mask].mean()) if inliers > 0 else float(residuals.mean())

    return {
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "avg_residual": avg_residual,
    }


def _refine_angle(points_a, points_b, initial_angle, inlier_threshold,
                  search_range_deg, step_deg, refine_iterations):

    best_angle = initial_angle
    best_geo = _geometric_validate(
        points_a, points_b, initial_angle, inlier_threshold, refine_iterations
    )
    best_inliers = best_geo["inliers"]

    # Collect scores across the search grid for possible interpolation
    angles_tested = []
    inliers_tested = []
    residuals_tested = []

    steps = int(np.ceil(search_range_deg / step_deg))
    for k in range(-steps, steps + 1):
        test_angle = _wrap_angle(initial_angle + k * step_deg)
        geo = _geometric_validate(
            points_a, points_b, test_angle, inlier_threshold, refine_iterations
        )
        angles_tested.append(test_angle)
        inliers_tested.append(geo["inliers"])
        residuals_tested.append(geo["avg_residual"])

        if geo["inliers"] > best_inliers:
            best_inliers = geo["inliers"]
            best_angle = test_angle
            best_geo = geo
        elif geo["inliers"] == best_inliers and geo["avg_residual"] < best_geo["avg_residual"]:
            best_angle = test_angle
            best_geo = geo

    # Parabolic sub-step interpolation around the best grid point.
    # Fit y = a*x^2 + b*x + c to the three points around the best,
    # then take the vertex position for higher angular precision.
    try:
        idx = angles_tested.index(best_angle)
        if 0 < idx < len(angles_tested) - 1:
            # Use inlier count as the "y" value to maximise
            y_left = inliers_tested[idx - 1]
            y_mid = inliers_tested[idx]
            y_right = inliers_tested[idx + 1]

            # Only interpolate if we have a proper peak (mid > both sides)
            if y_mid >= y_left and y_mid >= y_right and (y_left + y_right - 2 * y_mid) != 0:
                # Vertex of parabola through (-1, y_left), (0, y_mid), (1, y_right)
                offset = 0.5 * (y_left - y_right) / (y_left + y_right - 2 * y_mid)
                offset = np.clip(offset, -1.0, 1.0)
                interpolated_angle = _wrap_angle(best_angle + offset * step_deg)

                # Verify the interpolated angle is actually better (or equal)
                geo_interp = _geometric_validate(
                    points_a, points_b, interpolated_angle,
                    inlier_threshold, refine_iterations
                )
                if geo_interp["inliers"] >= best_inliers:
                    best_angle = interpolated_angle
                    best_geo = geo_interp
    except (ValueError, ZeroDivisionError):
        pass

    return best_angle, best_geo


def _spatial_support(points_a, points_b, angle_deg, inlier_threshold, grid_size=4, shape_a=None, shape_b=None):
    """Measure whether geometric inliers are spatially distributed rather than
    concentrated in one small repetitive structure. Uses both images and
    combines occupied-cell coverage with the number of supporting cells."""
    R = _rotation_matrix(angle_deg)
    rotated_a = points_a @ R.T
    translation = np.median(points_b - rotated_a, axis=0)
    residuals = np.linalg.norm(rotated_a + translation - points_b, axis=1)
    mask = residuals < inlier_threshold
    if not np.any(mask):
        return {"spatial_coverage": 0.0, "spatial_cells": 0, "spatial_concentration": 0.0}

    # Normalize using the actual image extents rather than the inlier
    # bounding box. The latter can make a tiny concentrated cluster look
    # artificially well-distributed.
    pa = points_a[mask]
    pb = points_b[mask]
    def coverage(pts, shape):
        if shape is None or len(shape) < 2:
            h = max(float(np.max(pts[:,1])) + 1.0, 1.0)
            w = max(float(np.max(pts[:,0])) + 1.0, 1.0)
        else:
            h, w = float(shape[0]), float(shape[1])
        uv = np.empty_like(pts, dtype=np.float64)
        uv[:,0] = np.clip(pts[:,0] / max(w, 1.0), 0.0, 0.999999)
        uv[:,1] = np.clip(pts[:,1] / max(h, 1.0), 0.0, 0.999999)
        cells = np.floor(uv * grid_size).astype(int)
        unique = np.unique(cells[:,0] * grid_size + cells[:,1])
        return len(unique) / float(grid_size * grid_size), len(unique)

    cov_a, cells_a = coverage(pa, shape_a)
    cov_b, cells_b = coverage(pb, shape_b)
    cov = 0.5 * (cov_a + cov_b)
    occupied = min(cells_a, cells_b)
    concentration = occupied / max(len(pa), 1)
    return {"spatial_coverage": float(cov), "spatial_cells": int(occupied),
            "spatial_concentration": float(concentration)}


def _dedupe_candidates(candidates: List[Dict[str, any]], cfg) -> List[Dict[str, any]]:
    tol = cfg["angle_cluster_tolerance_deg"]
    kept: List[Dict[str, any]] = []
    for cand in sorted(candidates, key=lambda c: c["inlier_ratio"], reverse=True):
        if any(_angular_distance(cand["angle"], k["angle"]) <= tol for k in kept):
            continue
        kept.append(cand)
    return kept


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _round_candidate(c: Dict[str, any]) -> Dict[str, any]:
    out = {
        "angle": round(float(c["angle"]), 2),
        "sources": sorted(set(c.get("sources", []))),
        "votes": round(float(c.get("votes", 0.0)), 2),
        "vote_frac": round(float(c.get("vote_frac", 0.0)), 4),
        "inliers": int(c["inliers"]),
        "inlier_ratio": round(float(c["inlier_ratio"]), 4),
        "avg_residual_px": round(float(c["avg_residual"]), 3),
        "is_flip_variant": bool(c.get("is_flip_variant", False)),
        "parent_angle": (round(float(c["parent_angle"]), 2)
                          if c.get("parent_angle") is not None else None),
    }
    if "refined" in c:
        out["refined"] = bool(c["refined"])
        out["refinement_delta_deg"] = round(float(c.get("refinement_delta_deg", 0.0)), 2)
    return out


def _round_candidates_for_output(candidates: List[Dict[str, any]],
                                  max_items: int = 6) -> List[Dict[str, any]]:
    return [_round_candidate(c) for c in candidates[:max_items]]


def _empty_result(reason: str, runtime: float, num_correspondences: int,
                   correspondence_source: str) -> Dict[str, any]:
    return {
        "angle": None,
        "confidence": 0.0,
        "supported": False,
        "hypotheses": [],
        "selected": None,
        "ambiguity_180": False,
        "severely_ambiguous": False,
        "flip_inlier_ratio": 0.0,
        "num_correspondences": num_correspondences,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
        "sources_used": [],
        "correspondence_source": correspondence_source,
        "reason": reason,
        "runtime_seconds": runtime,
    }
