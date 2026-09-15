#!/usr/bin/env python3
"""Smoke test for the SIH26166 repository.

Checks:
- Python runtime
- NumPy
- OpenCV
- AKAZE availability
- registration package imports
- required public pipeline functions

Run:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")
    return condition


def main() -> int:
    print("=" * 72)
    print("SIH26166 — Repository Smoke Test")
    print("=" * 72)

    failures = 0

    # NumPy
    try:
        import numpy as np

        if not check(
            "NumPy import",
            True,
            np.__version__,
        ):
            failures += 1
    except Exception as exc:
        check("NumPy import", False, str(exc))
        failures += 1

    # OpenCV
    cv2 = None

    try:
        import cv2

        detail = f"OpenCV {cv2.__version__}"
        if not check("OpenCV import", True, detail):
            failures += 1
    except Exception as exc:
        check("OpenCV import", False, str(exc))
        failures += 1

    if cv2 is not None:
        akaze_available = hasattr(cv2, "AKAZE_create")

        if not check(
            "AKAZE_create available",
            akaze_available,
            "required by the recovery path" if akaze_available
            else "upgrade OpenCV to a version providing AKAZE_create",
        ):
            failures += 1

        if akaze_available:
            try:
                detector = cv2.AKAZE_create()
                detector_ok = detector is not None
            except Exception:
                detector_ok = False

            if not check(
                "AKAZE detector creation",
                detector_ok,
            ):
                failures += 1

    # Registration package
    try:
        from backend.registration.pipeline import (
            adaptive_register,
            adaptive_register_with_rotation,
        )

        imports_ok = (
            callable(adaptive_register)
            and callable(adaptive_register_with_rotation)
        )

        if not check(
            "Registration pipeline import",
            imports_ok,
            "public pipeline entry points available",
        ):
            failures += 1

    except Exception as exc:
        check(
            "Registration pipeline import",
            False,
            str(exc),
        )
        failures += 1

    # Visualization
    try:
        from backend.registration.visualization import (
            save_inlier_visualization,
        )

        if not check(
            "Visualization import",
            callable(save_inlier_visualization),
        ):
            failures += 1

    except Exception as exc:
        check(
            "Visualization import",
            False,
            str(exc),
        )
        failures += 1

    print()
    print("=" * 72)

    if failures:
        print(f"SMOKE TEST FAILED — {failures} check(s) failed.")
        print("=" * 72)
        return 1

    print("SMOKE TEST PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
