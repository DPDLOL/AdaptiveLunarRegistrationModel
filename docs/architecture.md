# 🏗️ Repository Architecture

> [!NOTE]
> This document explains the structural organization of the repository and how components interact.
> For algorithm details, see [docs/algorithm](docs_algorithm.md).

---

## 1. High-Level Structure

The repository cleanly separates **six concerns:**

```
AdaptiveLunarRegistrationModel/
│
├── 📄  README.md, LICENSE, requirements*.txt, .gitignore
│
├── 🖥️  backend/              ← Web app + registration logic
│   ├── app.py                 ← Flask entry point
│   └── registration/          ← Core CV package
│       ├── __init__.py
│       ├── pipeline.py        ← Orchestration
│       ├── preprocessing.py   ← Image preparation
│       ├── orb_seed.py        ← ORB features & matching
│       ├── akaze_recovery.py  ← AKAZE recovery mechanism
│       ├── optical_flow.py    ← LK optical flow refinement
│       ├── geometry.py        ← Homography & geometry
│       ├── rotation.py        ← Rotation estimation
│       ├── validation.py      ← Multi-signal acceptance
│       └── visualization.py   ← Inlier visualizations
│
├── 🌐 frontend/              ← Browser interface
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── app.js
│
├── ⚙️  scripts/               ← CLI experiment tools
│   ├── run_registration.py
│   ├── run_batch.py
│   ├── benchmark.py
│   ├── summarize_benchmark.py
│   ├── plot_benchmark.py
│   └── smoke_test.py
│
├── 🧪 tests/                 ← Automated test suite
│   ├── conftest.py
│   └── test_*.py
│
├── 📊 benchmarks/            ← Benchmark configs & results
│   ├── README.md
│   └── pairs.example.csv
│
└── 📚 docs/                  ← Technical documentation
    ├── algorithm.md
    ├── architecture.md
    ├── development.md
    ├── evaluation_protocol.md
    ├── benchmark_comparison_template.md
    └── references.md
```

---

## 2. Backend Layer

### `backend/app.py` — Flask Application

The web-facing entry point with **strictly limited responsibilities:**

| Responsibility | Description |
|:--|:--|
| Image intake | Accept uploaded images |
| Decoding | Decode with OpenCV |
| Pipeline invocation | Call the public registration API |
| Visualization | Generate inlier visualizations |
| Response | Return structured JSON results |
| Static serving | Serve uploaded & generated images |

> [!IMPORTANT]
> The web layer should **not** contain registration algorithms directly.

### `backend/registration/` — Core Package

The actual registration implementation, organized by pipeline stage:

```
pipeline.py  ──── Orchestration
    │
    ├──▶ preprocessing.py     Image preparation
    ├──▶ orb_seed.py          Feature detection & matching
    ├──▶ akaze_recovery.py    Recovery mechanism
    ├──▶ optical_flow.py      Dense refinement
    ├──▶ geometry.py          Geometric utilities
    ├──▶ rotation.py          Rotation estimation
    ├──▶ validation.py        Acceptance logic
    └──▶ visualization.py     Result visualization
```

> `pipeline.py` decides **which path** is required for a given image pair. Supporting modules implement **individual stages**.

---

## 3. Frontend Layer

### `frontend/templates/index.html`

The HTML template provides:

- 📤 Two image upload controls
- ▶️ Registration action trigger
- ✅ Result status display
- 📊 Registration metrics table
- 🖼️ Inlier visualization

### `frontend/static/style.css`

Presentation and responsive layout rules.

### `frontend/static/app.js`

Browser-side behavior:

| Function | Purpose |
|:--|:--|
| Collect images | Gather the two uploaded files |
| Send request | Multipart POST to `/run` |
| Process response | Parse JSON registration results |
| Render metrics | Display in the metrics table |
| Show visualization | Display the inlier overlay |
| Error reporting | Surface failures to the user |

> [!TIP]
> The frontend does **not** implement any image-registration logic.

---

## 4. Script Layer

The `scripts/` directory contains **reproducible command-line utilities:**

| Script | Purpose |
|:--|:--|
| `run_registration.py` | Run a single image pair → console metrics + inlier visualization |
| `run_batch.py` | Run a CSV list of pairs → structured results CSV |
| `benchmark.py` | Repeated timing measurements → raw benchmark CSV |
| `summarize_benchmark.py` | Aggregate raw runs → one summary row per pair |
| `plot_benchmark.py` | Generate runtime visualization from summary |
| `smoke_test.py` | Verify dependencies & package availability |

---

## 5. Test Layer

The `tests/` directory provides automated checks via **pytest**.

| File | Verifies |
|:--|:--|
| `conftest.py` | Repository root available to pytest |
| `test_imports.py` | NumPy, OpenCV, AKAZE availability |
| `test_module_imports.py` | Registration package imports |
| `test_preprocessing.py` | Image preprocessing correctness |
| `test_geometry.py` | Geometric utility functions |
| `test_optical_flow.py` | Optical flow stage |
| `test_rotation.py` | Rotation estimation |
| `test_validation.py` | Validation logic |
| `test_visualization.py` | Visualization generation |
| `test_pipeline_smoke.py` | End-to-end smoke test |

> The test suite can be expanded with synthetic geometric tests, positive/negative pair tests, and regression tests for known failure cases.

---

## 6. Benchmark Layer

The `benchmarks/` directory contains benchmark instructions and result artifacts.

**Intended workflow:**

```
pairs.csv
    │
    ▼
run_batch.py ──▶ results.csv
    │
    ▼
benchmark.py ──▶ benchmark_runs.csv
    │
    ▼
summarize_benchmark.py ──▶ summary.csv
    │
    ▼
plot_benchmark.py ──▶ runtime.png
```

> Actual lunar datasets should remain **outside** the repository unless their redistribution is permitted.

---

## 7. Documentation Layer

The `docs/` directory contains technical documentation too detailed for the root README:

| Document | Contents |
|:--|:--|
| [`algorithm.md`](docs_algorithm.md) | Pipeline & validation logic |
| [`architecture.md`](docs_architecture.md) | Repository component interactions |
| [`development.md`](docs_development.md) | Setup, workflow, and contribution guide |
| [`evaluation_protocol.md`](docs_evaluation_protocol.md) | Benchmark methodology |
| [`benchmark_comparison_template.md`](docs_benchmark_comparison_template.md) | Reporting template for comparisons |
| [`references.md`](docs_references.md) | Papers, documentation & baselines |

---

## 8. Data Flow — Web Application

A browser request follows this path:

```
  🌐 Browser
      │
      │  multipart/form-data (two images)
      ▼
  🖥️ Flask /run endpoint
      │
      ▼
  📷 OpenCV image decode
      │
      ▼
  🧬 adaptive_register_with_rotation()
      │
      ├──▶ Fast ORB / MAGSAC / LK path
      ├──▶ Conditional recovery path
      ├──▶ Rotation handling (when applicable)
      └──▶ Final validation
      │
      ▼
  📦 Registration Result
      │
      ├──▶ JSON metrics
      └──▶ Inlier visualization
      │
      ▼
  🌐 Browser
```

---

## 9. Design Boundaries

A useful rule for maintaining the repository:

| Layer | Responsibility |
|:--|:--|
| **Frontend** | Presentation only |
| **Flask backend** | Transport / application boundary |
| **Registration package** | Computer-vision logic |
| **Scripts** | Reproducible execution & experiments |
| **Tests** | Correctness checks |
| **Benchmarks** | Measurement |
| **Docs** | Explanation |

> Keeping these boundaries clear makes the project easier to **test**, **benchmark**, **modify**, and **present** as a research prototype.

---

## 10. Research-Prototype Boundary

> [!WARNING]
> The repository is organized for **experimentation and reproducible evaluation**. It should not be described as flight-qualified software. Operational deployment would require additional validation, testing, resource characterization, fault handling, and system-level qualification.
