"""Geometric validation utilities for Adaptive Hybrid Lunar Registration."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

MIN_SEED_AREA_RATIO = 0.20
MAX_SEED_AREA_RATIO = 5.0
MAX_SEED_ANISOTROPY = 4.0
FINAL_MAGSAC = 1.25


def transform_sanity(H: Optional[np.ndarray], shape: Tuple[int, int]):
    if H is None or not np.all(np.isfinite(H)):
        return False, 0.0, math.inf
    h, w = shape
    corners = np.float32([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]])
    try:
        q = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return False, 0.0, math.inf
    if not np.all(np.isfinite(q)):
        return False, 0.0, math.inf
    x, y = q[:, 0], q[:, 1]
    area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5
    ref_area = max((w - 1) * (h - 1), 1)
    area_ratio = float(area / ref_area)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    pts = np.float32([[[cx, cy]], [[cx + 1, cy]], [[cx, cy + 1]]])
    try:
        qp = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        J = np.column_stack((qp[1] - qp[0], qp[2] - qp[0]))
        s = np.linalg.svd(J, compute_uv=False)
        anisotropy = math.inf if s[1] <= 1e-8 else float(s[0] / s[1])
    except Exception:
        anisotropy = math.inf
    sane = MIN_SEED_AREA_RATIO <= area_ratio <= MAX_SEED_AREA_RATIO and anisotropy <= MAX_SEED_ANISOTROPY
    return bool(sane), area_ratio, anisotropy


def grid_coverage(src: np.ndarray, shape: Tuple[int, int]) -> float:
    if len(src) == 0:
        return 0.0
    h, w = shape
    g = 4
    x = np.clip((src[:, 0] / max(w, 1) * g).astype(int), 0, g - 1)
    y = np.clip((src[:, 1] / max(h, 1) * g).astype(int), 0, g - 1)
    cells = len(set(zip(x.tolist(), y.tolist())))
    return cells / float(g * g)


def grid_cell_count(src: np.ndarray, shape: Tuple[int, int]) -> int:
    if len(src) == 0:
        return 0
    h, w = shape
    g = 4
    x = np.clip((src[:, 0] / max(w, 1) * g).astype(int), 0, g - 1)
    y = np.clip((src[:, 1] / max(h, 1) * g).astype(int), 0, g - 1)
    return len(set(zip(x.tolist(), y.tolist())))


def reproj_stats(H: Optional[np.ndarray], src: np.ndarray, dst: np.ndarray):
    if H is None or len(src) == 0:
        return math.inf, math.inf
    pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    e = np.linalg.norm(pred - dst, axis=1)
    return float(np.sqrt(np.mean(e * e))), float(np.median(e))


def final_geometry(src: np.ndarray, dst: np.ndarray, shape: Tuple[int, int]):
    empty_src = np.empty((0, 2), np.float32)
    empty_dst = np.empty((0, 2), np.float32)
    if len(src) < 10:
        return None, 0, math.inf, 0.0, 0, 0.0, math.inf, empty_src, empty_dst
    cv2.setRNGSeed(12345)
    H, mask = cv2.findHomography(
        src, dst, cv2.USAC_MAGSAC, FINAL_MAGSAC,
        maxIters=10000, confidence=0.999,
    )
    if H is None or mask is None:
        return None, 0, math.inf, 0.0, 0, 0.0, math.inf, empty_src, empty_dst
    m = mask.ravel().astype(bool)
    n = int(m.sum())
    if n == 0:
        return None, 0, math.inf, 0.0, 0, 0.0, math.inf, empty_src, empty_dst
    in_src, in_dst = src[m], dst[m]
    rms, _ = reproj_stats(H, in_src, in_dst)
    coverage = grid_coverage(in_src, shape)
    cells = grid_cell_count(in_src, shape)
    sane, area_ratio, anisotropy = transform_sanity(H, shape)
    return H, n, rms, coverage, cells, area_ratio, anisotropy, in_src.astype(np.float32), in_dst.astype(np.float32)
