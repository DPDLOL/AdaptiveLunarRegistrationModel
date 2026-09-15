"""Visualization helpers for Adaptive Hybrid Lunar Registration.

Preserves the tested inlier-visualization behavior used by the web frontend
and command-line runner.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _prep_visual_image(img: np.ndarray) -> np.ndarray:
    """Convert an input image to displayable 8-bit grayscale BGR."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if gray.dtype != np.uint8:
        lo, hi = np.percentile(gray, [1, 99])
        if hi <= lo:
            gray = cv2.convertScaleAbs(gray)
        else:
            x = (gray.astype(np.float32) - lo) * 255.0 / (hi - lo)
            gray = np.clip(x, 0, 255).astype(np.uint8)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def save_inlier_visualization(
    img1: np.ndarray,
    img2: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    out_path: Path,
    title: str,
    accepted: bool,
    max_draw: int = 300,
) -> None:
    """Save a side-by-side inlier correspondence image for every run."""
    left = _prep_visual_image(img1)
    right = _prep_visual_image(img2)

    # Keep output manageable for large lunar frames.
    max_h = 900
    scale = min(
        1.0,
        max_h / max(left.shape[0], right.shape[0]),
    )

    if scale < 1.0:
        new_left = (
            max(1, int(left.shape[1] * scale)),
            max(1, int(left.shape[0] * scale)),
        )
        new_right = (
            max(1, int(right.shape[1] * scale)),
            max(1, int(right.shape[0] * scale)),
        )

        left = cv2.resize(
            left,
            new_left,
            interpolation=cv2.INTER_AREA,
        )
        right = cv2.resize(
            right,
            new_right,
            interpolation=cv2.INTER_AREA,
        )

        src_draw = np.asarray(src_pts, np.float32) * scale
        dst_draw = np.asarray(dst_pts, np.float32) * scale
    else:
        src_draw = np.asarray(src_pts, np.float32).copy()
        dst_draw = np.asarray(dst_pts, np.float32).copy()

    canvas = np.zeros(
        (
            max(left.shape[0], right.shape[0]),
            left.shape[1] + right.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[:left.shape[0], :left.shape[1]] = left

    xoff = left.shape[1]
    canvas[:right.shape[0], xoff:] = right

    n = min(len(src_draw), len(dst_draw))

    if n:
        # Deterministic subsampling when there are thousands of inliers.
        idx = np.linspace(
            0,
            n - 1,
            min(n, max_draw),
        ).astype(int)

        rng = np.random.default_rng(12345)

        # Fixed shuffled ordering makes dense line fields easier to inspect.
        idx = idx[np.argsort(rng.random(len(idx)))]

        for i in idx:
            p1 = tuple(np.round(src_draw[i]).astype(int))
            p2 = tuple(np.round(dst_draw[i]).astype(int))
            p2_canvas = (p2[0] + xoff, p2[1])

            cv2.line(
                canvas,
                p1,
                p2_canvas,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                p1,
                2,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                p2_canvas,
                2,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )

    band_h = 48
    out = np.zeros(
        (canvas.shape[0] + band_h, canvas.shape[1], 3),
        dtype=np.uint8,
    )
    out[band_h:] = canvas

    status = "ACCEPTED" if accepted else "REJECTED"

    cv2.putText(
        out,
        f"{title} | {status} | FINAL INLIERS: {n}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
