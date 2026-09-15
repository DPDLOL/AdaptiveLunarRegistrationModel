#!/usr/bin/env python3
"""Summarize SIH26166 benchmark results.

Reads the CSV produced by scripts/benchmark.py and writes one summary row
per image pair.

Example:
    python scripts/summarize_benchmark.py benchmarks/results.csv \
        --output benchmarks/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SIH26166 benchmark results."
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Benchmark CSV produced by benchmark.py.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/summary.csv"),
        help="Summary CSV path.",
    )

    return parser.parse_args()


def to_float(value: str):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def mean_or_blank(values: list[float]):
    return statistics.mean(values) if values else ""


def median_or_blank(values: list[float]):
    return statistics.median(values) if values else ""


def summarize_group(rows: list[dict]) -> dict:
    wall_times = [
        value
        for value in (
            to_float(row.get("wall_time_s", ""))
            for row in rows
        )
        if value is not None
    ]

    runtimes = [
        value
        for value in (
            to_float(row.get("runtime_s", ""))
            for row in rows
        )
        if value is not None
    ]

    inliers = [
        value
        for value in (
            to_float(row.get("final_inliers", ""))
            for row in rows
        )
        if value is not None
    ]

    rms = [
        value
        for value in (
            to_float(row.get("final_rms", ""))
            for row in rows
        )
        if value is not None
    ]

    accepted = [
        str(row.get("accepted", "")).strip().lower() == "true"
        for row in rows
    ]

    accepted_count = sum(accepted)

    summary = {
        "image_a": rows[0].get("image_a", ""),
        "image_b": rows[0].get("image_b", ""),
        "runs": len(rows),
        "accepted_runs": accepted_count,
        "acceptance_rate": (
            accepted_count / len(rows)
            if rows
            else ""
        ),
        "mean_wall_time_s": mean_or_blank(wall_times),
        "median_wall_time_s": median_or_blank(wall_times),
        "min_wall_time_s": min(wall_times) if wall_times else "",
        "max_wall_time_s": max(wall_times) if wall_times else "",
        "mean_pipeline_runtime_s": mean_or_blank(runtimes),
        "mean_final_inliers": mean_or_blank(inliers),
        "mean_final_rms": mean_or_blank(rms),
    }

    if len(wall_times) > 1:
        summary["std_wall_time_s"] = statistics.stdev(wall_times)
    else:
        summary["std_wall_time_s"] = 0.0 if wall_times else ""

    return summary


def main() -> int:
    args = parse_args()

    if not args.input_csv.is_file():
        print(
            f"ERROR: Benchmark CSV not found: {args.input_csv}",
            file=sys.stderr,
        )
        return 2

    try:
        with args.input_csv.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except OSError as exc:
        print(f"ERROR: Could not read CSV: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print("ERROR: Benchmark CSV is empty.", file=sys.stderr)
        return 2

    required = {
        "image_a",
        "image_b",
    }

    missing = required - set(reader.fieldnames or [])
    if missing:
        print(
            "ERROR: Missing required columns: "
            + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
        return 2

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in rows:
        key = (
            row.get("image_a", ""),
            row.get("image_b", ""),
        )
        groups[key].append(row)

    summaries = [
        summarize_group(group)
        for group in groups.values()
    ]

    summaries.sort(
        key=lambda row: (
            str(row["image_a"]),
            str(row["image_b"]),
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_a",
        "image_b",
        "runs",
        "accepted_runs",
        "acceptance_rate",
        "mean_wall_time_s",
        "median_wall_time_s",
        "std_wall_time_s",
        "min_wall_time_s",
        "max_wall_time_s",
        "mean_pipeline_runtime_s",
        "mean_final_inliers",
        "mean_final_rms",
    ]

    try:
        with args.output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(summaries)
    except OSError as exc:
        print(f"ERROR: Could not write summary: {exc}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("SIH26166 — Benchmark Summary")
    print("=" * 72)
    print(f"Pairs summarized : {len(summaries)}")
    print(f"Output           : {args.output}")

    all_wall_times = [
        value
        for row in rows
        for value in [to_float(row.get("wall_time_s", ""))]
        if value is not None
    ]

    if all_wall_times:
        print(f"Overall mean     : {statistics.mean(all_wall_times):.4f}s")
        print(f"Overall median   : {statistics.median(all_wall_times):.4f}s")
        print(f"Overall minimum  : {min(all_wall_times):.4f}s")
        print(f"Overall maximum  : {max(all_wall_times):.4f}s")

    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
