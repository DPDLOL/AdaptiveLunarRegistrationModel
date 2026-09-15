#!/usr/bin/env python3
"""Batch runner for SIH26166 lunar image registration.

Input CSV format:

image_a,image_b
data/pairs/p1.png,data/pairs/p2.png
data/pairs/p3.png,data/pairs/p4.png

Example:
    python scripts/run_batch.py pairs.csv --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.registration.pipeline import adaptive_register_with_rotation


RESULT_FIELDS = [
    "image_a",
    "image_b",
    "accepted",
    "stage",
    "registration_mode",
    "rotation_used",
    "rotation_supported",
    "rotation_angle_deg",
    "rotation_confidence",
    "seed_inliers",
    "seed_coverage",
    "seed_area_ratio",
    "seed_anisotropy",
    "seed_sane",
    "recovery_used",
    "recovery_matches",
    "lk_points",
    "final_inliers",
    "final_rms",
    "final_coverage",
    "final_cells",
    "final_area_ratio",
    "final_anisotropy",
    "final_geometry_ok",
    "seed_runtime_s",
    "lk_runtime_s",
    "rotation_runtime_s",
    "runtime_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SIH26166 registration over image pairs."
    )

    parser.add_argument(
        "pairs_csv",
        type=Path,
        help="CSV containing image_a,image_b columns.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.csv"),
        help="Output CSV path (default: results.csv).",
    )

    return parser.parse_args()


def csv_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def read_pairs(path: Path) -> list[tuple[Path, Path]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pairs CSV not found: {path}")

    pairs = []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        required = {"image_a", "image_b"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain 'image_a' and 'image_b' columns."
            )

        for row_number, row in enumerate(reader, start=2):
            image_a = (row.get("image_a") or "").strip()
            image_b = (row.get("image_b") or "").strip()

            if not image_a or not image_b:
                raise ValueError(
                    f"Missing image path at CSV row {row_number}."
                )

            pairs.append(
                (
                    Path(image_a),
                    Path(image_b),
                )
            )

    return pairs


def run_pair(image_a: Path, image_b: Path) -> dict:
    result_row = {
        "image_a": str(image_a),
        "image_b": str(image_b),
    }

    img1 = cv2.imread(str(image_a), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(str(image_b), cv2.IMREAD_UNCHANGED)

    if img1 is None:
        result_row["stage"] = "IMAGE_A_DECODE_FAILED"
        result_row["accepted"] = False
        return result_row

    if img2 is None:
        result_row["stage"] = "IMAGE_B_DECODE_FAILED"
        result_row["accepted"] = False
        return result_row

    result = adaptive_register_with_rotation(img1, img2)

    for field in RESULT_FIELDS:
        if field in {"image_a", "image_b"}:
            continue

        result_row[field] = csv_value(result.get(field, ""))

    result_row["accepted"] = bool(result.get("accepted", False))

    return result_row


def main() -> int:
    args = parse_args()

    try:
        pairs = read_pairs(args.pairs_csv)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not pairs:
        print("ERROR: No image pairs found.", file=sys.stderr)
        return 2

    results = []

    print("=" * 72)
    print("SIH26166 — Batch Registration")
    print("=" * 72)
    print(f"Pairs : {len(pairs)}")
    print()

    for index, (image_a, image_b) in enumerate(pairs, start=1):
        print(
            f"[{index}/{len(pairs)}] "
            f"{image_a}  <->  {image_b}"
        )

        try:
            row = run_pair(image_a, image_b)
        except Exception as exc:
            row = {
                "image_a": str(image_a),
                "image_b": str(image_b),
                "accepted": False,
                "stage": "RUNTIME_ERROR",
            }
            print(f"  ERROR: {exc}")

        results.append(row)

        print(
            f"  accepted={row.get('accepted', False)} "
            f"stage={row.get('stage', '')} "
            f"inliers={row.get('final_inliers', '')} "
            f"rms={row.get('final_rms', '')} "
            f"runtime_s={row.get('runtime_s', '')}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)

    accepted_count = sum(
        bool(row.get("accepted", False))
        for row in results
    )

    print()
    print("=" * 72)
    print(f"Accepted : {accepted_count}/{len(results)}")
    print(f"Rejected : {len(results) - accepted_count}/{len(results)}")
    print(f"Results  : {args.output}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
