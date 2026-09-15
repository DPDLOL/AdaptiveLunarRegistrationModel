# 📊 Benchmark Comparison Template

> [!NOTE]
> This is a **reporting template** for comparing the Adaptive Hybrid Lunar Registration system against alternative methods.
> All numeric values **must** come from the actual benchmark experiment.

---

## 1. Primary Comparison

| Method | TPR | TNR | FAR | FRR | Mean Runtime (s) | P95 Runtime (s) |
|:--|--:|--:|--:|--:|--:|--:|
| **Adaptive Hybrid** | — | — | — | — | — | — |
| ORB + Geometric Est. | — | — | — | — | — | — |
| AKAZE + Geometric Est. | — | — | — | — | — | — |
| SURF + Geometric Est. | — | — | — | — | — | — |
| SuperPoint + SuperGlue | — | — | — | — | — | — |
| LoFTR | — | — | — | — | — | — |
| SuperPoint + LightGlue | — | — | — | — | — | — |

---

## 2. Registration Quality

| Method | Mean Inliers | Mean Inlier Ratio | Mean RMS (px) | Mean Coverage | Mean Cells | Mean Area Ratio | Mean Anisotropy |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Adaptive Hybrid** | — | — | — | — | — | — | — |
| ORB + Geometric Est. | — | — | — | — | — | — | — |
| AKAZE + Geometric Est. | — | — | — | — | — | — | — |
| SURF + Geometric Est. | — | — | — | — | — | — | — |
| SuperPoint + SuperGlue | — | — | — | — | — | — | — |
| LoFTR | — | — | — | — | — | — | — |
| SuperPoint + LightGlue | — | — | — | — | — | — | — |

---

## 3. Dataset Summary

> Fill in the exact dataset used for the reported comparison.

| Field | Value |
|:--|:--|
| Dataset name / version | |
| Number of positive pairs | |
| Number of negative pairs | |
| Number of difficult pairs | |
| Number of rotated pairs | |
| Image resolution range | |
| Ground-truth labeling method | |

---

## 4. Software & Hardware

| Field | Value |
|:--|:--|
| Repository commit | |
| Python version | |
| OpenCV version | |
| NumPy version | |
| Operating system | |
| CPU | |
| GPU | |
| RAM | |
| Baseline software / checkpoint versions | |
| Benchmark repeat count | |

---

## 5. Baseline Configuration

### Adaptive Hybrid

| Parameter | Value |
|:--|:--|
| Configuration | |
| Seed thresholds | |
| Final thresholds | |
| Recovery settings | |
| Rotation settings | |

### ORB Baseline

| Parameter | Value |
|:--|:--|
| Detector configuration | |
| Matcher configuration | |
| Geometric verification | |
| Acceptance rule | |

### AKAZE Baseline

| Parameter | Value |
|:--|:--|
| Detector configuration | |
| Matcher configuration | |
| Geometric verification | |
| Acceptance rule | |

### SURF Baseline

| Parameter | Value |
|:--|:--|
| Detector configuration | |
| Matcher configuration | |
| Geometric verification | |
| Acceptance rule | |

### Learned Baselines (SuperGlue / LoFTR / LightGlue)

| Parameter | Value |
|:--|:--|
| Model / checkpoint | |
| Input resolution | |
| Preprocessing | |
| Confidence threshold | |
| Geometric verification | |
| Acceptance rule | |

---

## 6. Per-Pair Results

> A detailed table should be retained in **CSV form**.

**Recommended fields:**

| Column | Description |
|:--|:--|
| `image_a` | Path to first image |
| `image_b` | Path to second image |
| `ground_truth` | `1` = same region, `0` = different |
| `method` | Registration method name |
| `accepted` | `true` / `false` |
| `stage` | Pipeline stage reached |
| `final_inliers` | Final inlier count |
| `inlier_ratio` | Inlier / total ratio |
| `final_rms` | RMS reprojection error (px) |
| `final_coverage` | Spatial coverage score |
| `final_cells` | Occupied 4×4 grid cells |
| `final_area_ratio` | Transformed area ratio |
| `final_anisotropy` | Directional distortion |
| `runtime_s` | Runtime in seconds |

---

## 7. Error Analysis

### 🔴 False Accepts

> A non-matching lunar pair incorrectly accepted.

| Field | Value |
|:--|:--|
| Pair | |
| Method | |
| Why it was accepted | |
| Inlier count | |
| Spatial coverage | |
| Transformation | |
| Observed failure mode | |

### 🟡 False Rejects

> A matching pair incorrectly rejected.

| Field | Value |
|:--|:--|
| Pair | |
| Method | |
| Why it was rejected | |
| Seed stage | |
| Recovery stage | |
| Final inliers | |
| Observed failure mode | |

### ⚠️ Degenerate Consensus

Cases where a high number of correspondences is concentrated in a **local image region**.

> Document the correspondence distribution and geometry metrics.

### 🔧 Recovery Failures

Cases where the adaptive system entered AKAZE or rotation recovery but **could not** establish a valid final registration.

---

## 8. Interpretation

> [!IMPORTANT]
> A method should **not** be declared superior from a single metric.

The final discussion should distinguish between:

| Dimension | Question |
|:--|:--|
| **Classification** | Does it correctly accept/reject pairs? |
| **Quality** | How accurate is the registration? |
| **Runtime** | How fast is it? |
| **Robustness** | Does it handle difficult cases? |
| **Failure modes** | How does it fail, and how badly? |

> Raw inlier count should **not** be interpreted independently of reprojection error, spatial coverage, and transformation validity.

---

## 9. Reproducibility Rule

> [!CAUTION]
> Every published number in a final comparison table must be traceable to:

```
  Dataset
     +
  Configuration
     +
  Software version
     +
  Hardware
     +
  Benchmark script
     +
  Raw result file
```

**Do not fill this template with estimated or assumed values.**
