# 🧬 Algorithm — Adaptive Hybrid Lunar Registration

> [!NOTE]
> This document provides a deep technical walkthrough of the registration pipeline.
> For a high-level overview, see the [main README](../README.md).

---

## 1. Objective

Given two lunar images:

1. **Determine** whether they depict the same lunar region.
2. **Estimate** a reliable geometric registration when the evidence supports it.

> **Central design principle:** *Make the system trust the right correspondences and reject the wrong ones.*

The pipeline is **adaptive** — it does not send every pair through the most expensive recovery path.

---

## 2. Pipeline Overview

```
  Image A + Image B
         │
         ▼
  Grayscale / 8-bit Normalization
         │
         ▼
  ┌──────────────────────────────────┐
  │   Two Structural Representations │
  │                                  │
  │   ├── CLAHE → Scharr            │
  │   └── CLAHE → Unsharp → Scharr  │
  └──────────────┬───────────────────┘
                 ▼
  ORB Feature Generation
         │
         ▼
  Correspondence Fusion / Deduplication
         │
         ▼
  Coarse USAC_MAGSAC Homography
         │
         ▼
  Seed-Quality Analysis
         │
    ┌────┴──────────────────┐
    │                       │
    ▼                       ▼
  ╔═══════════════╗   Seed Sanity Gate
  ║  CATASTROPHIC ║         │
  ║    REJECT     ║         ▼
  ╚═══════════════╝   H-seeded LK Optical Flow
                            │
                            ▼
                      Drift Filtering
                            │
                            ▼
                      Final MAGSAC
                            │
                            ▼
                  Spatial + Geometric Validation
                       ┌────┴────┐
                       ▼         ▼
                   ╔═══════╗ ╔════════╗
                   ║ACCEPT ║ ║ REJECT ║
                   ╚═══════╝ ╚════════╝


  ── Recovery Branch (weak but potentially recoverable) ──

  AKAZE Recovery → Guided Matching → LK Densification → Final Validation
```

---

## 3. Structural Preprocessing

Each input is converted to a stable **8-bit representation**.

Two complementary structural representations are generated:

| # | Pipeline | Purpose |
|:-:|:--|:--|
| 1 | CLAHE → Scharr | Gradient-based terrain structure |
| 2 | CLAHE → Unsharp → Scharr | Enhanced edges before gradient extraction |

> **Why two representations?**
> They emphasize terrain structure (crater rims, boundaries, local gradients) rather than raw intensity — *and* provide two partially independent sources of feature correspondences.

---

## 4. ORB Seed Stage

ORB is used as the **fast first-stage** detector and descriptor.

**Current implementation:**

- ORB features at a **reduced image scale**
- Separate feature sets from both structural representations
- Binary descriptor matching via **FLANN-LSH**
- **Fusion** of correspondences from both representations
- **Deduplication** before geometric estimation

**Output:** A coarse correspondence set.

---

## 5. Coarse MAGSAC Geometry

A coarse homography is estimated using OpenCV's **USAC_MAGSAC** estimator.

> [!IMPORTANT]
> The seed is **not** trusted merely because RANSAC-style estimation returned a model.

The seed is evaluated using **four independent signals:**

| Signal | What it catches |
|:--|:--|
| **Inlier count** | Insufficient evidence |
| **Spatial coverage** | Local-only consensus |
| **Area ratio** | Degenerate or impossible scaling |
| **Anisotropy** | Excessive directional distortion |

This is critical because a visually plausible local consensus can still be a **degenerate or incorrect** solution.

---

## 6. Catastrophic Seed Gate

Clearly pathological seeds are **rejected immediately**.

**Catastrophic conditions include:**

- ❌ Very small inlier support
- ❌ Extreme transformed area ratio
- ❌ Excessive anisotropy

> [!TIP]
> This is both a **correctness** and **runtime** design decision: an obviously wrong pair should not be given thousands of additional feature matches and flow tracks.

---

## 7. Seed Sanity Gate

A non-catastrophic seed is considered **usable** when it meets the configured sanity conditions:

- ✅ Minimum seed inliers
- ✅ Minimum seed spatial coverage
- ✅ Bounded area ratio
- ✅ Bounded anisotropy

| Seed Classification | Action |
|:--|:--|
| **Usable** | Proceed directly to optical-flow refinement |
| **Weak but plausible** | Enter recovery path |

---

## 8. AKAZE Recovery

AKAZE is used as a **secondary recovery mechanism** when the fast ORB seed is weak or degenerate — but **not catastrophically wrong**.

**Recovery workflow:**

```
Coarse geometric estimate
        │
        ▼
  Guided correspondence search
  (restricted to plausible region)
        │
        ▼
  Merge with existing evidence
        │
        ▼
  Refinement stage
```

> The recovery path is **intentionally conditional** rather than always-on.

---

## 9. H-Seeded Lucas-Kanade Refinement

The estimated homography predicts where source points should appear in the target image. Those predictions **initialize** pyramidal Lucas-Kanade optical flow.

**Current configuration:**

| Parameter | Setting |
|:--|:--|
| Window | Fixed size |
| Pyramid | Small image pyramid |
| Termination | Explicit iteration/convergence criterion |
| Point budget | Hard cap on tracked points |
| Point selection | Spatially balanced |
| Initialization | Flow seeded from homography |

A **drift filter** rejects tracks that move too far from the geometric prediction.

> **Result:** A relatively small feature correspondence set is turned into a **much denser** set of refined correspondences.

---

## 10. Final Robust Geometry

Refined correspondences pass through a **second MAGSAC homography** estimate — evaluated **independently** of the original seed.

**Final acceptance gate combines:**

- Final inlier count
- Final RMS reprojection error
- Spatial coverage
- Transformed area ratio
- Anisotropy

---

## 11. Spatial Validation

The image is divided into a **4×4 grid** (16 cells). A solution must cover **at least 8 cells**.

```
┌────┬────┬────┬────┐
│ ■  │ ■  │    │ ■  │
├────┼────┼────┼────┤
│ ■  │ ■  │ ■  │    │     ← Example: 10/16 cells occupied ✅
├────┼────┼────┼────┤
│    │ ■  │ ■  │ ■  │
├────┼────┼────┼────┤
│    │    │ ■  │    │
└────┴────┴────┴────┘
```

> Spatial distribution is a **first-class validation signal** — not a visualization-only statistic.

---

## 12. Geometry Validation

Two transformation sanity measures are especially important:

### Area Ratio

The transformed source region is compared with the original region.

```
Acceptance interval:  0.20 ≤ area_ratio ≤ 5.0
```

### Anisotropy

The transformation's directional distortion is monitored.

```
Acceptance limit:     anisotropy ≤ 4.0
```

> Together, these checks reject transformations that may have accumulated enough numerical inliers while still representing **implausible geometry**.

---

## 13. Final Acceptance Criteria — Summary

| Criterion | Threshold |
|:--|:--|
| Final inliers | ≥ 100 |
| Final RMS | ≤ 1.5 px |
| Spatial coverage | ≥ 0.50 (≥ 8 of 16 cells) |
| Area ratio | 0.20 … 5.0 |
| Anisotropy | ≤ 4.0 |

---

## 14. Rotation Handling

The geometry-driven rotation estimator generates candidate hypotheses from:

- 🔄 Feature-orientation differences
- 📐 Pairwise vector geometry
- 📏 Similarity-transform estimation
- 🎯 Baseline hypothesis

**Additional features:**
- **180° ambiguity handling** — the flipped variant is explicitly tested
- **Local angle refinement** — fine-tuning after clustering

> Rotation is treated as a **recovery mechanism** and should not alter a successful baseline registration path.

---

## 15. Design Rationale

The architecture deliberately follows this progression:

```
  Cheap evidence
      ↓
  Robust geometric seed
      ↓
  Trust / reject decision
      ↓
  Dense refinement
      ↓
  Strict final validation
```

The important property is **not** simply obtaining many correspondences. The system asks whether those correspondences are:

- [x] Spatially distributed
- [x] Geometrically consistent
- [x] Supported by a plausible transformation
- [x] Accurate enough under reprojection

This is the basis for the **adaptive trust/rejection strategy**.

---

## 16. Research Prototype Status

> [!WARNING]
> This implementation is a **research prototype**. Thresholds, runtime characteristics, and generalization behavior must be evaluated on a sufficiently large and representative lunar dataset before making claims of operational or flight readiness.
