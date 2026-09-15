"""Image preprocessing utilities for Adaptive Hybrid Lunar Registration.

This module preserves the preprocessing behavior of the validated research
prototype: 8-bit normalization, CLAHE, Scharr gradient magnitude, and
common-shape padding.
"""

from typing import Tuple

import cv2
import numpy as np

def gray8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return cv2.convertScaleAbs(img)
    x = (img.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(x, 0, 255).astype(np.uint8)


def structural_reps(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """CLAHE+Scharr and CLAHE+Unsharp+Scharr."""
    img = gray8(img)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    eq = clahe.apply(img)

    blur = cv2.GaussianBlur(eq, (0, 0), 3.0)
    sharp = cv2.addWeighted(eq, 1.5, blur, -0.5, 0)

    def scharr_mag(x: np.ndarray) -> np.ndarray:
        gx = cv2.Scharr(x, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(x, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        p1, p99 = np.percentile(mag, [1, 99])
        if p99 <= p1:
            return np.zeros_like(x, dtype=np.uint8)
        mag = (mag - p1) * 255.0 / (p99 - p1)
        return np.clip(mag, 0, 255).astype(np.uint8)

    return scharr_mag(eq), scharr_mag(sharp)


def pad_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])

    def pad(x: np.ndarray) -> np.ndarray:
        if x.shape[:2] == (h, w):
            return x
        out = np.zeros((h, w), dtype=x.dtype)
        out[:x.shape[0], :x.shape[1]] = x
        return out

    return pad(a), pad(b)

