# 📋 Evaluation Protocol

> [!NOTE]
> This document defines the measurement methodology for benchmarking the Adaptive Hybrid Lunar Registration system and comparing it against baselines.

---

## 1. Purpose

The benchmark should answer **two separate questions:**

| # | Question | Measures |
|:-:|:--|:--|
| 1 | **Correctness** | Does the system accept true matches and reject false matches? |
| 2 | **Efficiency** | How much computation is required to reach that decision? |

> The **same** image-pair set and measurement rules should be used for the adaptive system and every baseline.

---

## 2. Dataset Composition

The evaluation set should contain **labeled** image pairs:

| Label | Meaning |
|:--|:--|
| `1` (positive) | Same lunar region |
| `0` (negative) | Different lunar regions |

**The set should include representative variation:**

- 🌗 Illumination differences
- 🔍 Scale changes
- 📐 Viewpoint changes
- 🧩 Partial overlap
- 🏜️ Difficult texture / weak feature regions
- 🔄 Rotated cases
- 🪤 Visually similar but incorrect regions

> [!CAUTION]
> Do **not** claim broad generalization from a very small dataset.

---

## 3. Required Labels

Every benchmark pair needs a ground-truth label.

**Example CSV format:**

```csv
image_a,image_b,label
data/pairs/p1.png,data/pairs/p2.png,1
data/pairs/p3.png,data/pairs/p4.png,0
```

> The label file should be kept **synchronized** with the pair list.

---

## 4. Primary Classification Metrics

For each method, calculate:

### True Positive Rate (TPR)

```
TPR = TP / (TP + FN)
```

*How many true matching pairs are correctly **accepted**.*

### True Negative Rate (TNR)

```
TNR = TN / (TN + FP)
```

*How many non-matching pairs are correctly **rejected**.*

### False Accept Rate (FAR)

```
FAR = FP / (FP + TN)
```

> [!WARNING]
> FAR is especially important for a registration system — **accepting a wrong lunar region is more dangerous** than failing to register a valid pair.

### False Reject Rate (FRR)

```
FRR = FN / (FN + TP)
```

*Valid pairs incorrectly rejected.*

---

## 5. Registration-Quality Metrics

For **accepted** registrations, record:

| Metric | Why It Matters |
|:--|:--|
| Final inlier count | Evidence strength |
| Inlier ratio | Signal-to-noise |
| RMS reprojection error | Geometric accuracy |
| Spatial coverage | Distribution quality |
| Occupied 4×4 cells | Coverage granularity |
| Area ratio | Transformation plausibility |
| Anisotropy | Directional distortion |

> [!IMPORTANT]
> A large inlier count should **not** be reported without the corresponding geometric and spatial checks.

---

## 6. Runtime Metrics

For every method, record:

| Statistic | Purpose |
|:--|:--|
| Mean runtime | Average case |
| Median runtime | Typical case |
| Min / Max runtime | Range bounds |
| Standard deviation | Variability |
| **P95 runtime** | Worst-case tail |

**Rules:**
- Use the **same hardware and software** for methods being compared.
- Measure **complete** inference/registration execution — not only an isolated internal function (unless explicitly stated).

---

## 7. Warm-Up and Repetition

For timing experiments, follow this protocol:

| Step | Action |
|:-:|:--|
| 1 | Load all required models and dependencies **before** timed runs |
| 2 | Perform a **warm-up** run when the method requires initialization |
| 3 | Run each pair **multiple** times |
| 4 | Record **every** run |
| 5 | Report **aggregate** statistics |

> The exact repeat count should be stated in the benchmark report.

---

## 8. Baseline Fairness

> [!IMPORTANT]
> When comparing with another registration system, ensure a **level playing field.**

**Checklist for fair comparison:**

- [x] Same image pairs
- [x] Same positive/negative labels
- [x] Document preprocessing
- [x] Document image scaling
- [x] Document matching thresholds
- [x] Document geometric verification thresholds
- [x] Document model weights (when applicable)
- [x] Document hardware
- [x] Document software versions

**Rules:**
- Do **not** silently tune one method using information unavailable to others.
- If a baseline **cannot process** a particular pair, record that failure — don't remove the pair.

---

## 9. Decision Threshold Handling

Some methods produce a confidence score rather than a direct accept/reject decision.

**Acceptable thresholding approaches:**

| Approach | Description |
|:--|:--|
| Fixed threshold | From published / default settings |
| Validation-tuned | Selected on a **separate** validation set |

> [!CAUTION]
> Do **not** tune thresholds directly on the final test set and then report the result as an unbiased evaluation.

---

## 10. Reporting Table

**Template for final comparison:**

| Method | TPR | TNR | FAR | FRR | Mean Runtime | P95 Runtime | Mean Inliers | Mean RMS |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **Adaptive Hybrid** | — | — | — | — | — | — | — | — |
| ORB baseline | — | — | — | — | — | — | — | — |
| AKAZE baseline | — | — | — | — | — | — | — | — |
| SURF baseline | — | — | — | — | — | — | — | — |
| SuperPoint + SuperGlue | — | — | — | — | — | — | — | — |
| LoFTR | — | — | — | — | — | — | — | — |
| SuperPoint + LightGlue | — | — | — | — | — | — | — | — |

> Dashes (`—`) indicate values that **must** be obtained from the actual benchmark — they are **not** placeholder performance claims.

---

## 11. Error Analysis

> Aggregate metrics alone are **not sufficient**.

For rejected positives and accepted negatives, inspect:

- 🖼️ Inlier visualization
- 📍 Spatial distribution
- 📐 Transformation geometry
- 🔄 Rotation hypothesis
- 🌱 Seed quality
- 🔧 Recovery path
- 📏 Final reprojection residuals

**Key failure categories to document:**

| Category | Description |
|:--|:--|
| **False accept** | Wrong image pair accepted |
| **False reject** | Same region incorrectly rejected |
| **Degenerate consensus** | High inlier count but poor spatial coverage |
| **Geometric distortion** | Implausible area ratio or anisotropy |
| **Recovery failure** | Plausible pair enters recovery but can't produce a valid model |

---

## 12. Reproducibility Record

Every reported benchmark should record:

| Field | Example |
|:--|:--|
| Repository commit | `a1b2c3d` |
| Python version | `3.11.5` |
| OpenCV version | `4.10.0` |
| NumPy version | `1.26.4` |
| Operating system | `Ubuntu 22.04` |
| CPU / hardware | `AMD Ryzen 9 7950X` |
| Dataset version | `lunar_pairs_v3` |
| Pair-list version | `pairs_20260915.csv` |
| Benchmark repeat count | `5` |
| Baseline configuration | *(method-specific)* |
| Model/checkpoint versions | *(for learned methods)* |

> This makes later comparison tables **traceable** to a specific experiment.

---

## 13. Interpretation

The adaptive system should **not** be judged only by raw feature-match count.

**Intended evaluation hierarchy:**

```
  Correct pair?
      │
      ▼
  Reliable geometric model?
      │
      ▼
  Spatially distributed support?
      │
      ▼
  Plausible transformation?
      │
      ▼
  ✅ Accept
```

> The benchmark should consider **both** classification performance **and** registration quality.

---

## 14. Research-Prototype Boundary

> [!WARNING]
> A favorable benchmark result does **not** establish flight readiness. Operational claims would require substantially broader validation — including representative datasets, robustness testing, resource characterization, failure handling, and system-level verification.
