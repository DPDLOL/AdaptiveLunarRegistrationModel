#!/usr/bin/env python3
"""Benchmark the SIH26166 registration pipeline.

The benchmark repeats each image-pair registration a configurable number of
times and reports aggregate runtime statistics.

Example:
    python scripts/benchmark.py pairs.csv --repeats 5 --output benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.registration.pipeline import adaptive_register_with_rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SIH26166 registration runtime."
    )

    parser.add_argument(
        "pairs_csv",
        type=Path,
        help="CSV containing image_a,image_b columns.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs per image pair (default: 3).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results.csv"),
        help="CSV path for per-run timings.",
    )

    return parser.parse_args()


def load_pairs(path: Path) -> list[tuple[Path, Path]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError("CSV has no header.")

        if "image_a" not in reader.fieldnames or "image_b" not in reader.fieldnames:
            raise ValueError(
                "CSV must contain 'image_a' and 'image_b' columns."
            )

        pairs = []

        for row in reader:
            image_a = Path((row.get("image_a") or "").strip())
            image_b = Path((row.get("image_b") or "").strip())

            if image_a and image_b:
                pairs.append((image_a, image_b))

    return pairs


def main() -> int:
    args = parse_args()

    if args.repeats < 1:
        print("ERROR: --repeats must be >= 1.", file=sys.stderr)
        return 2

    if not args.pairs_csv.is_file():
        print(
            f"ERROR: Pair CSV not found: {args.pairs_csv}",
            file=sys.stderr,
        )
        return 2

    try:
        pairs = load_pairs(args.pairs_csv)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not pairs:
        print("ERROR: No image pairs found.", file=sys.stderr)
        return 2

    rows = []
    all_runtimes = []

    print("=" * 72)
    print("SIH26166 — Registration Benchmark")
    print("=" * 72)
    print(f"Pairs   : {len(pairs)}")
    print(f"Repeats : {args.repeats}")
    print()

    for pair_index, (image_a, image_b) in enumerate(pairs, start=1):
        if not image_a.is_file():
            print(f"WARNING: Missing {image_a}; skipping.")
            continue

        if not image_b.is_file():
            print(f"WARNING: Missing {image_b}; skipping.")
            continue

        img1 = cv2.imread(str(image_a), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(str(image_b), cv2.IMREAD_UNCHANGED)

        if img1 is None or img2 is None:
            print(
                f"WARNING: Could not decode pair "
                f"{image_a} <-> {image_b}; skipping."
            )
            continue

        pair_times = []

        print(
            f"[{pair_index}/{len(pairs)}] "
            f"{image_a.name} <-> {image_b.name}"
        )

        for repeat in range(1, args.repeats + 1):
            start = time.perf_counter()

            result = adaptive_register_with_rotation(img1, img2)

            elapsed = time.perf_counter() - start
            pair_times.append(elapsed)
            all_runtimes.append(elapsed)

            rows.append(
                {
                    "pair_index": pair_index,
                    "repeat": repeat,
                    "image_a": str(image_a),
                    "image_b": str(image_b),
                    "accepted": bool(result.get("accepted", False)),
                    "stage": result.get("stage", ""),
                    "final_inliers": result.get("final_inliers", ""),
                    "final_rms": result.get("final_rms", ""),
                    "runtime_s": result.get("runtime_s", ""),
                    "wall_time_s": elapsed,
                }
            )

            print(
                f"  run {repeat}: "
                f"{elapsed:.4f}s wall, "
                f"accepted={result.get('accepted', False)}"
            )

        print(
            f"  mean={statistics.mean(pair_times):.4f}s "
            f"min={min(pair_times):.4f}s "
            f"max={max(pair_times):.4f}s"
        )
        print()

    if not rows:
        print("ERROR: No benchmark runs were completed.", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_index",
                "repeat",
                "image_a",
                "image_b",
                "accepted",
                "stage",
                "final_inliers",
                "final_rms",
                "runtime_s",
                "wall_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("Aggregate benchmark")
    print("=" * 72)
    print(f"Runs completed : {len(all_runtimes)}")
    print(f"Mean           : {statistics.mean(all_runtimes):.4f}s")
    print(f"Median         : {statistics.median(all_runtimes):.4f}s")
    print(f"Minimum        : {min(all_runtimes):.4f}s")
    print(f"Maximum        : {max(all_runtimes):.4f}s")
    print(f"Std. deviation : {statistics.stdev(all_runtimes):.4f}s"
          if len(all_runtimes) > 1
          else "Std. deviation : 0.0000s")
    print(f"Results        : {args.output}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
