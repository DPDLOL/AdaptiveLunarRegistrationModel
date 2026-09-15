"""Lucas-Kanade optical-flow refinement for lunar registration.

Preserves the tested prototype stage:
spatially balanced FAST points -> homography initialization ->
pyramidal Lucas-Kanade -> 3 px drift filtering.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

FAST_THRESHOLD = 10
MAX_FAST_POINTS = 5000

LK_WIN = (11, 11)
LK_LEVEL = 2
LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    10,
    0.03,
)

MAX_DRIFT = 3.0


def select_fast_points(img: np.ndarray, budget: int) -> np.ndarray:
    fast = cv2.FastFeatureDetector_create(
        threshold=FAST_THRESHOLD,
        nonmaxSuppression=True,
    )
    keypoints = fast.detect(img, None)

    if not keypoints:
        return np.empty((0, 1, 2), np.float32)

    if len(keypoints) <= budget:
        return np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)

    h, w = img.shape[:2]
    grid = 4
    per_cell = max(1, budget // (grid * grid))
    buckets = [[] for _ in range(grid * grid)]

    for kp in keypoints:
        x, y = kp.pt
        cx = min(grid - 1, max(0, int(x / max(w, 1) * grid)))
        cy = min(grid - 1, max(0, int(y / max(h, 1) * grid)))
        buckets[cy * grid + cx].append(kp)

    selected = []
    for bucket in buckets:
        bucket.sort(key=lambda kp: kp.response, reverse=True)
        selected.extend(bucket[:per_cell])

    selected_ids = {id(kp) for kp in selected}
    remaining = [kp for kp in keypoints if id(kp) not in selected_ids]
    remaining.sort(key=lambda kp: kp.response, reverse=True)
    selected.extend(remaining[:max(0, budget - len(selected))])
    selected = selected[:budget]

    return np.float32([kp.pt for kp in selected]).reshape(-1, 1, 2)


def run_lk(
    img1: np.ndarray,
    img2: np.ndarray,
    H: np.ndarray,
    budget: int,
) -> Tuple[np.ndarray, np.ndarray]:
    p0 = select_fast_points(img1, budget)

    if len(p0) < 10:
        empty = np.empty((0, 1, 2), np.float32)
        return empty, empty

    # Homography provides the initial destination for each source point.
    guess = cv2.perspectiveTransform(p0, H)

    h, w = img2.shape[:2]
    x = guess[:, 0, 0]
    y = guess[:, 0, 1]

    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )

    p0 = p0[inside]
    guess = guess[inside]

    if len(p0) < 10:
        empty = np.empty((0, 1, 2), np.float32)
        return empty, empty

    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        img1,
        img2,
        p0,
        guess,
        winSize=LK_WIN,
        maxLevel=LK_LEVEL,
        criteria=LK_CRITERIA,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )

    if p1 is None or status is None:
        empty = np.empty((0, 1, 2), np.float32)
        return empty, empty

    drift = np.linalg.norm(p1 - guess, axis=2).ravel()

    valid = (
        (status.ravel() == 1)
        & np.isfinite(drift)
        & (drift <= MAX_DRIFT)
    )

    return p0[valid], p1[valid]
