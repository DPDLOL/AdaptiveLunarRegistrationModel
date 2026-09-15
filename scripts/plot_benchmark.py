#!/usr/bin/env python3
"""Plot SIH26166 benchmark runtime summary.

Reads the summary CSV produced by scripts/summarize_benchmark.py and creates
a simple horizontal bar chart of mean wall-clock runtime per image pair.

Example:
    python scripts/plot_benchmark.py benchmarks/summary.csv \
        --output benchmarks/runtime.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SIH26166 benchmark runtime results."
    )

    parser.add_argument(
        "summary_csv",
        type=Path,
        help="Summary CSV produced by summarize_benchmark.py.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/runtime.png"),
        help="Output image path (default: benchmarks/runtime.png).",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.summary_csv.is_file():
        print(
            f"ERROR: Summary CSV not found: {args.summary_csv}",
            file=sys.stderr,
        )
        return 2

    labels = []
    runtimes = []

    try:
        with args.summary_csv.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)

            required = {
                "image_a",
                "image_b",
                "mean_wall_time_s",
            }

            missing = required - set(reader.fieldnames or [])
            if missing:
                print(
                    "ERROR: Missing required columns: "
                    + ", ".join(sorted(missing)),
                    file=sys.stderr,
                )
                return 2

            for row in reader:
                try:
                    runtime = float(row["mean_wall_time_s"])
                except (TypeError, ValueError):
                    continue

                labels.append(
                    f"{Path(row['image_a']).name}\n"
                    f"↔ {Path(row['image_b']).name}"
                )
                runtimes.append(runtime)

    except OSError as exc:
        print(f"ERROR: Could not read CSV: {exc}", file=sys.stderr)
        return 2

    if not runtimes:
        print("ERROR: No valid benchmark rows found.", file=sys.stderr)
        return 2

    height = max(4.0, 0.55 * len(runtimes) + 1.5)

    fig, ax = plt.subplots(figsize=(11, height))

    positions = list(range(len(runtimes)))

    ax.barh(positions, runtimes)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Mean Wall-Clock Runtime (s)")
    ax.set_title(
        "SIH26166 — Adaptive Hybrid Lunar Registration Runtime"
    )

    ax.grid(axis="x", alpha=0.25)

    for position, runtime in zip(positions, runtimes):
        ax.text(
            runtime,
            position,
            f"  {runtime:.3f}s",
            va="center",
        )

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(args.output, dpi=160, bbox_inches="tight")
    except OSError as exc:
        print(f"ERROR: Could not write plot: {exc}", file=sys.stderr)
        return 2
    finally:
        plt.close(fig)

    print(f"Runtime plot written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
