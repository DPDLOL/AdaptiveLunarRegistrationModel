# 📊 Benchmarking

> [!NOTE]
> This directory contains scripts and outputs used to evaluate the Adaptive Hybrid Lunar Registration pipeline on a common set of lunar image pairs.

---

## 📁 Recommended Dataset Organization

Keep the actual image dataset **outside** the source tree when possible (especially for large collections).

**Suggested layout:**

```
project-root/
│
├── data/                        ← Outside Git (large datasets)
│   └── pairs/
│       ├── p1.png
│       ├── p2.png
│       ├── p3.png
│       └── p4.png
│
└── benchmarks/                  ← Inside Git (configs & results)
    ├── pairs.csv
    ├── results.csv
    ├── summary.csv
    └── runtime.png
```

**Pair-list format (CSV):**

```csv
image_a,image_b
data/pairs/p1.png,data/pairs/p2.png
data/pairs/p3.png,data/pairs/p4.png
```

> Include both **positive and negative** pairs. Difficult and rotated cases should also be represented.

---

## ▶️ Batch Evaluation

Run the full registration pipeline over the pair list:

```bash
python scripts/scripts_run_batch.py benchmarks/pairs.csv \
    --output benchmarks/results.csv
```

Records the registration decision and main geometric metrics for every pair.

---

## ⏱️ Runtime Benchmarking

For repeated timing measurements:

```bash
python scripts/scripts_benchmark.py benchmarks/pairs.csv \
    --repeats 5 \
    --output benchmarks/benchmark_runs.csv
```

Records both the **pipeline-reported runtime** and the **wall-clock time** measured by the benchmark harness.

---

## 📈 Summary Generation

Convert raw benchmark runs into one row per image pair:

```bash
python scripts/scripts_summarize_benchmark.py \
    benchmarks/benchmark_runs.csv \
    --output benchmarks/summary.csv
```

---

## 🎨 Runtime Plot

Generate a runtime visualization:

```bash
python scripts/scripts_plot_benchmark.py \
    benchmarks/summary.csv \
    --output benchmarks/runtime.png
```

---

## 📏 Metrics

### Per-Pair Metrics

| Category | Metrics |
|:--|:--|
| **Decision** | Acceptance / rejection, pipeline stage |
| **Rotation** | Rotation usage, rotation support |
| **Seed quality** | Inliers, spatial coverage, area ratio, anisotropy |
| **Recovery** | Recovery usage, recovery matches |
| **Refinement** | LK tracked points |
| **Final** | Inliers, RMS error, spatial coverage, occupied cells, area ratio, anisotropy |
| **Timing** | Runtime (seconds) |

### Dataset-Level Comparison Metrics

| Category | Metrics |
|:--|:--|
| **Classification** | TPR, TNR, FAR, FRR |
| **Quality** | Inlier count, inlier ratio, RMS error, spatial coverage |
| **Runtime** | Mean, median, P95 |

---

## 🏁 Comparison Baselines

The repository can be extended with baseline implementations:

| Category | Methods |
|:--|:--|
| **Classical** | ORB, SIFT, AKAZE, SURF + geometric estimation |
| **Learned** | SuperPoint + SuperGlue, LoFTR, SuperPoint + LightGlue |

> Every baseline should be evaluated on the **same image pairs** and under the **same measurement protocol**.

---

## 🔒 Reproducibility

When reporting benchmarks, record:

| Field | Description |
|:--|:--|
| Image-pair list | Exact CSV used |
| OpenCV version | e.g., `4.10.0` |
| Python version | e.g., `3.11.5` |
| Dependency versions | NumPy, Flask, etc. |
| Benchmark repeat count | e.g., `5` |
| Machine / CPU info | When reporting runtime |
| Baseline-specific config | Thresholds, model weights, etc. |

> [!CAUTION]
> Do **not** commit large proprietary or restricted lunar datasets to the repository. Store dataset instructions or download information in the documentation instead.

---

## ⚠️ Research Status

> Benchmark numbers should be generated from the **actual evaluation dataset**. This directory does not contain invented or placeholder performance claims.
