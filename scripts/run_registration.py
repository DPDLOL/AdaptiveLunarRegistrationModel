#!/usr/bin/env python3
"""Run SIH26166 Adaptive Hybrid Lunar Image Registration from the command line.

Example:
    python scripts/run_registration.py image_a.png image_b.png

The script runs the same public registration pipeline used by the Flask
application and optionally writes an inlier visualization.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np


# Make the repository root importable when this file is executed directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.registration.pipeline import adaptive_register_with_rotation
from backend.registration.visualization import save_inlier_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register two lunar images using SIH26166."
    )

    parser.add_argument(
        "image_a",
        type=Path,
        help="Path to the first lunar image.",
    )

    parser.add_argument(
        "image_b",
        type=Path,
        help="Path to the second lunar image.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated visualization output "
             "(default: outputs).",
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Do not save the inlier visualization.",
    )

    return parser.parse_args()


def safe_value(value):
    """Convert NumPy/scalar values into readable JSON-safe values."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def main() -> int:
    args = parse_args()

    image_a = args.image_a
    image_b = args.image_b

    if not image_a.is_file():
        print(f"ERROR: Image A not found: {image_a}", file=sys.stderr)
        return 2

    if not image_b.is_file():
        print(f"ERROR: Image B not found: {image_b}", file=sys.stderr)
        return 2

    img1 = cv2.imread(str(image_a), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(str(image_b), cv2.IMREAD_UNCHANGED)

    if img1 is None:
        print(f"ERROR: Could not decode Image A: {image_a}", file=sys.stderr)
        return 2

    if img2 is None:
        print(f"ERROR: Could not decode Image B: {image_b}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("SIH26166 — Adaptive Hybrid Lunar Image Registration")
    print("=" * 72)
    print(f"Image A : {image_a}")
    print(f"Image B : {image_b}")
    print(f"Shape A : {img1.shape}")
    print(f"Shape B : {img2.shape}")
    print()

    result = adaptive_register_with_rotation(img1, img2)

    accepted = bool(result.get("accepted", False))

    print(f"Accepted             : {accepted}")
    print(f"Stage                : {result.get('stage', '-')}")
    print(f"Registration mode    : {result.get('registration_mode', '-')}")
    print(f"Rotation used        : {result.get('rotation_used', '-')}")
    print(f"Rotation supported   : {result.get('rotation_supported', '-')}")
    print(f"Rotation angle (deg) : {result.get('rotation_angle_deg', '-')}")
    print(f"Rotation confidence  : {result.get('rotation_confidence', '-')}")
    print(f"Seed inliers         : {result.get('seed_inliers', '-')}")
    print(f"Seed coverage        : {result.get('seed_coverage', '-')}")
    print(f"Seed area ratio      : {result.get('seed_area_ratio', '-')}")
    print(f"Seed anisotropy      : {result.get('seed_anisotropy', '-')}")
    print(f"Seed sane            : {result.get('seed_sane', '-')}")
    print(f"Recovery used        : {result.get('recovery_used', '-')}")
    print(f"Recovery matches     : {result.get('recovery_matches', '-')}")
    print(f"LK points            : {result.get('lk_points', '-')}")
    print(f"Final inliers        : {result.get('final_inliers', '-')}")
    print(f"Final RMS (px)       : {result.get('final_rms', '-')}")
    print(f"Final coverage       : {result.get('final_coverage', '-')}")
    print(f"Final cells          : {result.get('final_cells', '-')}")
    print(f"Final area ratio     : {result.get('final_area_ratio', '-')}")
    print(f"Final anisotropy     : {result.get('final_anisotropy', '-')}")
    print(f"Final geometry OK    : {result.get('final_geometry_ok', '-')}")
    print(f"Runtime (s)          : {result.get('runtime_s', '-')}")

    if not args.no_visualization:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        output_name = (
            f"inliers_{image_a.stem}__{image_b.stem}.png"
        )
        output_path = args.output_dir / output_name

        inlier_src = result.get(
            "inlier_src",
            np.empty((0, 2), dtype=np.float32),
        )
        inlier_dst = result.get(
            "inlier_dst",
            np.empty((0, 2), dtype=np.float32),
        )

        save_inlier_visualization(
            img1,
            img2,
            inlier_src,
            inlier_dst,
            output_path,
            f"{image_a.name}  vs  {image_b.name}",
            accepted,
        )

        print(f"Inlier visualization : {output_path}")

    print("=" * 72)

    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
