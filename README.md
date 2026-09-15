<p align="center">
  <h1 align="center">🌙 Adaptive Hybrid Lunar Image Registration</h1>
  <p align="center">
    <strong>SIH26166 — Space Technology / Software</strong>
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/-Quick_Start-0d1117?style=for-the-badge&logo=rocket&logoColor=white" alt="Quick Start"></a>
    <a href="docs/docs_algorithm.md"><img src="https://img.shields.io/badge/-Algorithm-0d1117?style=for-the-badge&logo=bookstack&logoColor=white" alt="Algorithm"></a>
    <a href="docs/docs_architecture.md"><img src="https://img.shields.io/badge/-Architecture-0d1117?style=for-the-badge&logo=blueprint&logoColor=white" alt="Architecture"></a>
    <a href="benchmarks/benchmarks_README.md"><img src="https://img.shields.io/badge/-Benchmarks-0d1117?style=for-the-badge&logo=speedtest&logoColor=white" alt="Benchmarks"></a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/opencv-4.10+-5c3ee8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
  </p>
</p>

---

An adaptive computer-vision pipeline that determines whether two lunar images correspond to the **same region** and, when they do, estimates a reliable **geometric registration** between them.

> **Core principle:** *Make the system trust the right correspondences and reject the wrong ones.*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Components](#-key-components)
- [Final Acceptance Criteria](#-final-acceptance-criteria)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Why Adaptive?](#-why-adaptive)
- [Validation Philosophy](#-validation-philosophy)
- [Benchmarking](#-benchmarking)
- [References](#-references)

---

## 🔭 Overview

The prototype combines **lightweight feature matching**, **robust geometric estimation**, **optical-flow refinement**, **recovery matching**, **rotation handling**, and **spatial/geometric validation** into a single adaptive pipeline.

### Pipeline at a Glance

```
                        ┌─────────────────────┐
                        │   Two Lunar Images   │
                        └──────────┬──────────┘
                                   ▼
                      ┌────────────────────────┐
                      │ Grayscale / 8-bit Norm  │
                      └────────────┬───────────┘
                                   ▼
               ┌───────────────────────────────────┐
               │     Structural Preprocessing       │
               │  ┌──────────────┬────────────────┐ │
               │  │ CLAHE+Scharr │ CLAHE+Unsharp  │ │
               │  │              │   +Scharr      │ │
               │  └──────────────┴────────────────┘ │
               └───────────────┬───────────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │ ORB Correspondence Gen.   │
                  └────────────┬─────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │  Correspondence Fusion    │
                  └────────────┬─────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │  Coarse MAGSAC Homography │
                  └────────────┬─────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │   Seed-Quality Analysis   │
                  └──────┬──────────────┬────┘
                         │              │
                    catastrophic    usable / weak
                         │              │
                         ▼              ▼
                      ╔═══════╗   ┌──────────┐
                      ║REJECT ║   │ usable?  │
                      ╚═══════╝   └──┬───┬───┘
                                  yes│   │no (recoverable)
                                     │   ▼
                                     │  ┌────────────────┐
                                     │  │ AKAZE Recovery  │
                                     │  └───────┬────────┘
                                     │          │
                                     ▼          ▼
                            ┌───────────────────────────┐
                            │ H-seeded LK Optical Flow   │
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │     Drift Filtering        │
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │      Final MAGSAC          │
                            └─────────────┬─────────────┘
                                          ▼
                            ┌───────────────────────────┐
                            │ Spatial + Geometric Valid.  │
                            └──────┬────────────────┬────┘
                                   │                │
                              ╔════╧════╗     ╔═════╧═════╗
                              ║ ACCEPT  ║     ║  REJECT   ║
                              ╚═════════╝     ╚═══════════╝
```

---

## 🧩 Key Components

<details>
<summary><strong>🔬 Structural Preprocessing</strong></summary>

Images are converted to a stable 8-bit representation and processed using **two structural representations**:

| Representation | Pipeline |
|:--|:--|
| **Gradient** | CLAHE → Scharr gradient magnitude |
| **Sharpened Gradient** | CLAHE → Unsharp sharpening → Scharr gradient magnitude |

> **Goal:** Emphasize terrain structure (crater rims, local boundaries) while reducing sensitivity to raw illumination differences.

</details>

<details>
<summary><strong>🎯 ORB Coarse Correspondence Stage</strong></summary>

ORB serves as the **fast first-stage** feature detector/descriptor. Two structural representations are matched independently, and their correspondences are **fused** before geometric estimation.

</details>

<details>
<summary><strong>📐 MAGSAC Geometric Seed</strong></summary>

A coarse homography is estimated using OpenCV's `USAC_MAGSAC`. The seed is evaluated across **four independent signals:**

| Signal | Purpose |
|:--|:--|
| Inlier count | Sufficient evidence |
| Spatial coverage | Avoids local-only consensus |
| Area ratio | Detects degenerate scaling |
| Anisotropy | Detects directional distortion |

</details>

<details>
<summary><strong>🚫 Catastrophic-Seed Fail-Fast Rejection</strong></summary>

Clearly pathological seeds are **rejected immediately** — preventing obviously wrong pairs from entering the expensive recovery path. This is both a correctness and a runtime design decision.

</details>

<details>
<summary><strong>🔄 AKAZE Recovery</strong></summary>

When the ORB seed is **weak or degenerate** but still potentially recoverable, AKAZE is used as a secondary feature/recovery mechanism. Guided matching restricts the search region using the coarse transformation.

</details>

<details>
<summary><strong>🔍 H-Seeded Lucas-Kanade Refinement</strong></summary>

The coarse homography predicts where source points should appear in the target image. **Pyramidal Lucas-Kanade optical flow** is initialized from those predictions to refine a much larger set of correspondences.

A drift limit rejects tracks that move too far from the geometric prediction.

</details>

<details>
<summary><strong>✅ Final Robust Geometry</strong></summary>

Refined correspondences pass through a **second MAGSAC homography estimation**. The final solution is accepted only when it satisfies all required criteria.

</details>

<details>
<summary><strong>🔀 Rotation Handling</strong></summary>

A geometry-driven rotation estimator generates hypotheses from multiple sources:

- Feature-orientation differences
- Pairwise vector geometry
- Similarity-transform estimation
- Baseline hypothesis

Candidate angles are clustered and geometrically validated. The implementation explicitly tests the **180°-flipped variant** and performs local angle refinement.

> Rotation is treated as an **additional recovery mechanism** — it never replaces a successful normal registration path.

</details>

---

## 🎯 Final Acceptance Criteria

The current prototype requires **all** of the following to accept a registration:

| Criterion | Threshold | Rationale |
|:--|:--|:--|
| Final inliers | **≥ 100** | Sufficient correspondence evidence |
| RMS reprojection error | **≤ 1.5 px** | Sub-pixel geometric accuracy |
| Spatial coverage | **≥ 8 / 16 cells** | Avoids degenerate local consensus |
| Area ratio | **0.20 – 5.0** | Rejects implausible scale changes |
| Anisotropy | **≤ 4.0** | Rejects directional distortion |

> **Why the 4×4 grid matters:** A very large local consensus can still be degenerate if all correspondences are concentrated in one small part of the image. Spatial distribution is a **first-class validation signal**.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd AdaptiveLunarRegistrationModel

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Command-Line Usage

```bash
python scripts/scripts_run_registration.py image_A.png image_B.png
```

### Web Interface

```bash
python -m backend.app
```

Then open **http://127.0.0.1:5000** — upload two lunar images and get structured JSON results with an inlier visualization.

---

## 📂 Repository Structure

```
AdaptiveLunarRegistrationModel/
│
├── 📄 README.md                    ← You are here
├── 📄 LICENSE                      ← MIT License
├── 📄 requirements.txt             ← Runtime dependencies
├── 📄 requirements-dev.txt         ← Dev/benchmark dependencies
├── 📄 .gitignore
│
├── 🖥️  backend/
│   ├── app.py                      ← Flask web application
│   └── registration/
│       ├── pipeline.py             ← Orchestration logic
│       ├── preprocessing.py        ← CLAHE / Scharr / normalization
│       ├── orb_seed.py             ← ORB feature generation & matching
│       ├── akaze_recovery.py       ← AKAZE recovery mechanism
│       ├── optical_flow.py         ← H-seeded Lucas-Kanade refinement
│       ├── geometry.py             ← Homography & geometric utilities
│       ├── rotation.py             ← Rotation estimation & handling
│       ├── validation.py           ← Multi-signal acceptance logic
│       └── visualization.py        ← Inlier visualization generation
│
├── 🌐 frontend/
│   ├── templates/                  ← HTML (Jinja2)
│   └── static/                     ← CSS + JS
│
├── ⚙️  scripts/                     ← CLI tools for experiments
├── 🧪 tests/                       ← Automated test suite (pytest)
├── 📊 benchmarks/                  ← Benchmark configs & results
└── 📚 docs/                        ← In-depth technical documentation
```

---

## 🧠 Why Adaptive?

The system does **not** run every image pair through the most expensive path:

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  🟢 Strong seed    → Fast ORB → LK path                     │
│                      (cheapest, most common)                 │
│                                                              │
│  🟡 Weak seed      → AKAZE / Rotation recovery → continue   │
│                      (moderate cost, only when justified)    │
│                                                              │
│  🔴 Pathological   → Reject early                            │
│                      (no wasted computation)                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

> **Computation is spent only when the evidence justifies it.**

---

## 🛡️ Validation Philosophy

The project does **not** treat raw match count as sufficient evidence. A registration is trusted only when **several independent signals agree:**

```
  Correspondence count          ─┐
  Low reprojection error        ─┤
  Broad spatial distribution    ─┼──▶  ✅  Accepted Registration
  Reasonable transformation     ─┤
  geometry                      ─┘
```

---

## 📊 Benchmarking

<details>
<summary><strong>Recommended Baselines</strong></summary>

| Category | Methods |
|:--|:--|
| **Classical** | ORB, SIFT, AKAZE, SURF + geometric estimation |
| **Learned** | SuperPoint + SuperGlue, LoFTR, SuperPoint + LightGlue |

</details>

<details>
<summary><strong>Recommended Metrics</strong></summary>

| Category | Metrics |
|:--|:--|
| **Classification** | TPR, TNR, FAR, FRR |
| **Quality** | Inlier count, inlier ratio, RMS reprojection error, spatial coverage |
| **Efficiency** | Mean runtime, P95 runtime |

</details>

> The benchmark should contain both **positive and negative** lunar image pairs, including **difficult** and **rotated** cases.
>
> 📖 See [benchmarks/README](benchmarks/benchmarks_README.md) and [docs/evaluation_protocol](docs/docs_evaluation_protocol.md) for full details.

---

## 📝 Output Metrics

For each registration run, the system reports:

<details>
<summary>Full output field list</summary>

| Field | Description |
|:--|:--|
| `accepted` | Registration decision |
| `stage` | Pipeline stage reached |
| `rotation_estimate` | Estimated angle & confidence |
| `seed_inliers` | Coarse-stage inlier count |
| `seed_coverage` | Seed spatial coverage |
| `seed_area_ratio` | Seed transformed-area ratio |
| `seed_anisotropy` | Seed directional distortion |
| `akaze_recovery` | Whether AKAZE recovery was used |
| `recovery_matches` | Matches from recovery stage |
| `lk_tracked` | Lucas-Kanade tracked points |
| `final_inliers` | Final inlier count |
| `final_rms` | Final RMS reprojection error |
| `final_coverage` | Final spatial coverage |
| `occupied_cells` | Occupied 4×4 grid cells |
| `final_area_ratio` | Final transformed-area ratio |
| `final_anisotropy` | Final directional distortion |
| `runtime` | Total pipeline runtime |

</details>

An **inlier visualization** can also be generated showing the final accepted correspondence set.

---

## 📚 References

| # | Paper | Venue |
|:-:|:--|:--|
| 1 | Rublee et al., **ORB: An Efficient Alternative to SIFT or SURF** | ICCV 2011 |
| 2 | Barath et al., **MAGSAC: Marginalizing Sample Consensus** | CVPR 2019 |
| 3 | Barath et al., **MAGSAC++, a Fast, Reliable and Accurate Robust Estimator** | CVPR 2020 |
| 4 | Lucas & Kanade, **An Iterative Image Registration Technique…** | IJCAI 1981 |
| 5 | Alcantarilla et al., **Fast Explicit Diffusion for Accelerated Features…** | BMVC 2013 |
| 6 | OpenCV documentation — CLAHE, Scharr, homography estimation, pyramidal LK optical flow | — |

---

## ⚠️ Research Status

> [!WARNING]
> This repository contains a **research prototype**. Thresholds and benchmark results should be evaluated on a sufficiently large and representative lunar dataset before making claims of general superiority or operational readiness.

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built for <strong>SIH26166</strong> — Space Technology / Software</sub>
</p>
