"""AKAZE recovery stage for Adaptive Hybrid Lunar Registration.

Extracted from the tested prototype without changing the recovery logic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


AKAZE_THRESHOLD = 5e-5
GUIDED_RADIUS = 3.0
GUIDED_RATIO = 0.90
MIN_LOCAL_CANDIDATES = 2
COARSE_MAGSAC = 3.0
RNG_SEED = 12345


def build_bins(points: np.ndarray, cell: float) -> Dict[Tuple[int, int], List[int]]:
    """Build spatial bins for fast local candidate lookup."""
    bins: Dict[Tuple[int, int], List[int]] = {}
    c = max(cell, 1.0)
    for i, (x, y) in enumerate(points):
        k = (int(x // c), int(y // c))
        bins.setdefault(k, []).append(i)
    return bins


def guided_akaze(
    rep1: np.ndarray,
    rep2: np.ndarray,
    H: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Recover AKAZE correspondences around H-predicted locations.

    The implementation matches the tested prototype:
    AKAZE detection -> H prediction -> 3 px spatial gating ->
    Hamming ratio test -> one-to-one target usage.
    """
    ak = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)

    kp1, des1 = ak.detectAndCompute(rep1, None)
    kp2, des2 = ak.detectAndCompute(rep2, None)

    if des1 is None or des2 is None:
        return []

    p1 = np.float32([k.pt for k in kp1])
    p2 = np.float32([k.pt for k in kp2])

    # Work in full-resolution coordinates.
    predicted = cv2.perspectiveTransform(
        p1.reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    bins = build_bins(p2, GUIDED_RADIUS)
    used = set()
    result: List[Tuple[np.ndarray, np.ndarray]] = []

    r2 = GUIDED_RADIUS ** 2

    for i, q in enumerate(predicted):
        if not np.all(np.isfinite(q)):
            continue

        bx, by = int(q[0] // GUIDED_RADIUS), int(q[1] // GUIDED_RADIUS)
        candidates: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(bins.get((bx + dx, by + dy), []))

        local = [
            j for j in candidates
            if np.sum((p2[j] - q) ** 2) <= r2
        ]

        if len(local) < MIN_LOCAL_CANDIDATES:
            continue

        dists = sorted(
            (
                float(cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)),
                j,
            )
            for j in local
        )

        d0, j0 = dists[0]
        d1 = dists[1][0]

        if d1 <= 0 or d0 >= GUIDED_RATIO * d1:
            continue
        if j0 in used:
            continue

        used.add(j0)
        result.append((p1[i], p2[j0]))

    return result


def global_akaze_bootstrap(
    rep1a: np.ndarray,
    rep2a: np.ndarray,
    rep1b: np.ndarray,
    rep2b: np.ndarray,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Optional[np.ndarray]]:
    """Run the small global AKAZE bootstrap used when no seed H exists.

    Returns the recovered pairs and a coarse MAGSAC homography when enough
    matches are available.
    """
    recovery_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    ak = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)

    for a, b in ((rep1a, rep2a), (rep1b, rep2b)):
        k1, d1 = ak.detectAndCompute(a, None)
        k2, d2 = ak.detectAndCompute(b, None)
        if d1 is None or d2 is None:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = bf.knnMatch(d1, d2, k=2)
        good = [
            m for pair in raw
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.82 * n.distance
        ]

        for m in good:
            recovery_pairs.append(
                (
                    np.float32(k1[m.queryIdx].pt),
                    np.float32(k2[m.trainIdx].pt),
                )
            )

    if len(recovery_pairs) < 10:
        return recovery_pairs, None

    s = np.float32([p[0] for p in recovery_pairs])
    d = np.float32([p[1] for p in recovery_pairs])

    cv2.setRNGSeed(RNG_SEED + 2)
    H2, mask2 = cv2.findHomography(
        s.reshape(-1, 1, 2),
        d.reshape(-1, 1, 2),
        cv2.USAC_MAGSAC,
        COARSE_MAGSAC,
        maxIters=10000,
        confidence=0.999,
    )

    if H2 is None:
        return recovery_pairs, None

    return recovery_pairs, H2


def guided_recovery_bootstrap(
    rep1a: np.ndarray,
    rep2a: np.ndarray,
    rep1b: np.ndarray,
    rep2b: np.ndarray,
    H: np.ndarray,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Optional[np.ndarray]]:
    """Run guided AKAZE on both representations and bootstrap H with MAGSAC."""
    guided_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    guided_pairs.extend(guided_akaze(rep1a, rep2a, H))
    guided_pairs.extend(guided_akaze(rep1b, rep2b, H))

    if len(guided_pairs) < 10:
        return guided_pairs, None

    s = np.float32([p[0] for p in guided_pairs])
    d = np.float32([p[1] for p in guided_pairs])

    cv2.setRNGSeed(RNG_SEED + 1)
    H2, mask2 = cv2.findHomography(
        s.reshape(-1, 1, 2),
        d.reshape(-1, 1, 2),
        cv2.USAC_MAGSAC,
        COARSE_MAGSAC,
        maxIters=10000,
        confidence=0.999,
    )

    return guided_pairs, H2 if H2 is not None and mask2 is not None else None
