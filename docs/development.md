# 🛠️ Development Guide

> [!NOTE]
> This guide covers environment setup, development workflow, and best practices.
> For algorithm details, see [docs/algorithm](docs_algorithm.md).

---

## 1. Environment Setup

### 📋 Prerequisites

- Python 3.10+
- pip
- Git

### 🪟 Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 🐧 Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 📦 Install Dependencies

```bash
# Runtime dependencies
pip install -r requirements.txt

# Development & benchmarking dependencies
pip install -r requirements-dev.txt
```

### ✅ Verify Installation

```bash
python scripts/scripts_smoke_test.py
```

---

## 2. Run the Web Application

Start Flask from the repository root:

```bash
python -m backend.app
```

Then open: **http://127.0.0.1:5000**

> The browser interface accepts two images and displays the registration decision, metrics, and inlier visualization.

---

## 3. Run a Single Image Pair (CLI)

```bash
python scripts/scripts_run_registration.py image_A.png image_B.png
```

By default, the script writes an inlier visualization to `outputs/`.

**Disable visualization:**

```bash
python scripts/scripts_run_registration.py image_A.png image_B.png --no-visualization
```

---

## 4. Run a Batch

**Step 1 — Create a pair-list CSV:**

```csv
image_a,image_b
data/pairs/p1.png,data/pairs/p2.png
data/pairs/p3.png,data/pairs/p4.png
```

**Step 2 — Execute:**

```bash
python scripts/scripts_run_batch.py benchmarks/pairs.csv \
    --output benchmarks/results.csv
```

---

## 5. Run Benchmarks

### ⏱️ Repeated Timing

```bash
python scripts/scripts_benchmark.py benchmarks/pairs.csv \
    --repeats 5 \
    --output benchmarks/benchmark_runs.csv
```

### 📊 Per-Pair Summary

```bash
python scripts/scripts_summarize_benchmark.py \
    benchmarks/benchmark_runs.csv \
    --output benchmarks/summary.csv
```

### 📈 Runtime Plot

```bash
python scripts/scripts_plot_benchmark.py \
    benchmarks/summary.csv \
    --output benchmarks/runtime.png
```

---

## 6. Run Tests

```bash
pytest
```

The test suite verifies:

| Test File | Coverage |
|:--|:--|
| `test_imports.py` | Core dependency availability |
| `test_module_imports.py` | Registration package imports |
| `test_preprocessing.py` | Image preprocessing |
| `test_geometry.py` | Geometric utilities |
| `test_optical_flow.py` | Optical flow stage |
| `test_rotation.py` | Rotation estimation |
| `test_validation.py` | Validation logic |
| `test_visualization.py` | Visualization generation |
| `test_pipeline_smoke.py` | End-to-end smoke test |

> As the project evolves, regression tests should be added for important positive, negative, difficult, and rotated cases.

---

## 7. Development Workflow

Recommended workflow for algorithm changes:

```
  ① Make the change
         │
         ▼
  ② Run smoke_test.py          ← Quick sanity check
         │
         ▼
  ③ Run pytest                 ← Full test suite
         │
         ▼
  ④ Run known pos/neg pairs    ← Regression check
         │
         ▼
  ⑤ Run the benchmark set      ← Performance impact
         │
         ▼
  ⑥ Inspect inlier visuals     ← Visual quality check
         │
         ▼
  ⑦ Update docs if needed      ← Keep docs in sync
```

> [!IMPORTANT]
> Do **not** treat an increase in raw match count as proof of improvement. Changes should be judged using the **full acceptance and validation logic**.

---

## 8. Reproducibility

When reporting an experiment, record:

| Field | Example |
|:--|:--|
| Python version | `3.11.5` |
| OpenCV version | `4.10.0` |
| NumPy version | `1.26.4` |
| Operating system | `Windows 11` / `Ubuntu 22.04` |
| CPU | `AMD Ryzen 9 7950X` |
| Image-pair list | `benchmarks/pairs_v2.csv` |
| Benchmark repeats | `5` |
| Algorithm config | `seed_min_inliers=15, …` |
| Result files | `benchmarks/results_20260915.csv` |

> [!TIP]
> This is particularly important when comparing runtime across machines.

---

## 9. Dataset Handling

> [!CAUTION]
> Large lunar datasets should remain **outside** the Git repository unless redistribution is explicitly permitted.

Use paths in benchmark CSV files to reference local datasets.

**Do NOT commit:**

| ❌ | Reason |
|:--|:--|
| Large image collections | Too large for Git |
| Generated inlier images | Reproducible from source |
| Benchmark result dumps | Machine-specific |
| Temporary uploads | Transient |
| Private/restricted mission data | Access-controlled |

> The repository should contain **reproducible instructions** rather than unlicensed or unnecessarily large datasets.

---

## 10. Algorithm Changes — Where Things Go

The registration package is deliberately separated into modules:

| What | Where |
|:--|:--|
| Web transport logic | `backend/app.py` |
| Registration logic | `backend/registration/` |
| CLI experiment logic | `scripts/` |
| Automated checks | `tests/` |
| Benchmark artifacts | `benchmarks/` |
| Technical explanations | `docs/` |

> This separation prevents experimental code from being mixed into the application boundary.

---

## 11. Research-Prototype Discipline

> [!WARNING]
> This repository is a **research prototype**.
>
> - Do **not** describe a local benchmark improvement as universal superiority.
> - **Report** the dataset, conditions, thresholds, and limitations alongside experimental numbers.
> - Any claim of operational or flight readiness requires **additional system-level validation** beyond this repository.
