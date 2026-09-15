"""ORB correspondence generation and coarse MAGSAC seed estimation.

Extracted from the tested Adaptive Hybrid Lunar Registration prototype without
changing the matching or coarse geometric-estimation logic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import grid_coverage, transform_sanity


# Prototype configuration.
ORB_FEATURES = 4000
ORB_RATIO = 0.80
ORB_SCALE = 1.2
ORB_LEVELS = 8
ORB_FAST_THRESHOLD = 10
COARSE_MAGSAC = 3.0
RNG_SEED = 12345


def orb_correspondences(
    img1: np.ndarray,
    img2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Generate ORB correspondences using the prototype's 0.5x path."""
    small1 = cv2.resize(
        img1,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_AREA,
    )
    small2 = cv2.resize(
        img2,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_AREA,
    )

    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=ORB_SCALE,
        nlevels=ORB_LEVELS,
        fastThreshold=ORB_FAST_THRESHOLD,
    )

    kp1, des1 = orb.detectAndCompute(small1, None)
    kp2, des2 = orb.detectAndCompute(small2, None)

    if des1 is None or des2 is None:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            {"kp1": 0, "kp2": 0, "raw": 0},
        )

    # FLANN LSH for ORB's binary descriptors.
    flann = cv2.FlannBasedMatcher(
        dict(
            algorithm=6,
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        ),
        dict(checks=50),
    )

    raw = flann.knnMatch(des1, des2, k=2)
    good = [
        m
        for pair in raw
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < ORB_RATIO * n.distance
    ]

    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])

    # Convert 0.5x coordinates to full resolution.
    src *= 2.0
    dst *= 2.0

    return src, dst, {
        "kp1": len(kp1),
        "kp2": len(kp2),
        "raw": len(good),
    }


def coarse_hypothesis(
    reps1: List[np.ndarray],
    reps2: List[np.ndarray],
    seed_offset: int = 0,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Fuse the two structural ORB representations and estimate a coarse H."""
    all_src: List[np.ndarray] = []
    all_dst: List[np.ndarray] = []

    stats = {
        "kp1": 0,
        "kp2": 0,
        "matches": 0,
        "seed_inliers": 0,
        "coverage": 0.0,
        "area_ratio": 0.0,
        "anisotropy": np.inf,
    }

    for rep1, rep2 in zip(reps1, reps2):
        s, d, st = orb_correspondences(rep1, rep2)
        stats["kp1"] += st["kp1"]
        stats["kp2"] += st["kp2"]
        stats["matches"] += st["raw"]

        if len(s):
            all_src.append(s)
            all_dst.append(d)

    if not all_src:
        return None, stats

    src = np.vstack(all_src)
    dst = np.vstack(all_dst)

    # Dedupe identical coordinate correspondences.
    keys = {}
    for a, b in zip(src, dst):
        k = (
            round(float(a[0]), 2),
            round(float(a[1]), 2),
            round(float(b[0]), 2),
            round(float(b[1]), 2),
        )
        keys[k] = (a, b)

    vals = list(keys.values())
    src = np.float32([x[0] for x in vals])
    dst = np.float32([x[1] for x in vals])

    if len(src) < 10:
        return None, stats

    cv2.setRNGSeed(RNG_SEED + seed_offset)
    H, mask = cv2.findHomography(
        src.reshape(-1, 1, 2),
        dst.reshape(-1, 1, 2),
        cv2.USAC_MAGSAC,
        COARSE_MAGSAC,
        maxIters=10000,
        confidence=0.999,
    )

    if H is None or mask is None:
        return None, stats

    m = mask.ravel().astype(bool)
    stats["seed_inliers"] = int(m.sum())
    stats["coverage"] = grid_coverage(src[m], reps1[0].shape)
    sane, area_ratio, anis = transform_sanity(H, reps1[0].shape)
    stats["area_ratio"] = area_ratio
    stats["anisotropy"] = anis
    stats["seed_sane"] = sane

    return H, stats
