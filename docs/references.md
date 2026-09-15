# 📖 References

> This document lists the external papers, OpenCV documentation, project-specific contributions, and candidate benchmark systems that inform the implementation.

---

## 📑 Feature Detection & Description

### ORB

> Rublee, E., Rabaud, V., Konolige, K., and Bradski, G.
>
> **ORB: An Efficient Alternative to SIFT or SURF.**
>
> *International Conference on Computer Vision (ICCV), 2011.*
>
> 🔗 [IEEE Xplore](https://ieeexplore.ieee.org/document/6126544)

**Usage in this project:** Fast first-stage correspondence generation.

---

### AKAZE

> Alcantarilla, P. F., Nuevo, J., and Bartoli, A.
>
> **Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces.**
>
> *British Machine Vision Conference (BMVC), 2013.*
>
> 🔗 [BMVA Archive](https://www.bmva-archive.org.uk/bmvc/2013/Papers/paper0013/index.html)

**Usage in this project:** Conditional recovery mechanism for weak but potentially recoverable seeds.

---

## 📐 Robust Geometric Estimation

### MAGSAC

> Barath, D., Matas, J., and Noskova, J.
>
> **MAGSAC: Marginalizing Sample Consensus.**
>
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.*
>
> 🔗 [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Barath_MAGSAC_Marginalizing_Sample_Consensus_CVPR_2019_paper.html)

**Usage in this project:** OpenCV's USAC/MAGSAC family for coarse and final homography estimation.

---

### MAGSAC++

> Barath, D., Noskova, J., Ivashechkin, M., and Matas, J.
>
> **MAGSAC++, a Fast, Reliable and Accurate Robust Estimator.**
>
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.*
>
> 🔗 [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html)

**Usage in this project:** Background for the robust-estimation family used by the pipeline.

---

## 🔍 Optical Flow & Image Registration

### Lucas-Kanade

> Lucas, B. D. and Kanade, T.
>
> **An Iterative Image Registration Technique with an Application to Stereo Vision.**
>
> *International Joint Conference on Artificial Intelligence (IJCAI), 1981.*
>
> 🔗 [IJCAI Proceedings (PDF)](https://www.ijcai.org/Proceedings/81-2/Papers/017.pdf)

**Usage in this project:** Pyramidal Lucas-Kanade optical flow for dense correspondence refinement after geometric seeding.

---

### OpenCV Optical-Flow Documentation

> OpenCV documentation for pyramidal Lucas-Kanade optical flow.
>
> 🔗 [OpenCV Tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)

---

## 🖼️ Image Preprocessing

### CLAHE

> OpenCV documentation for Contrast Limited Adaptive Histogram Equalization.
>
> 🔗 [OpenCV API](https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html)

**Usage in this project:** Part of the structural preprocessing representations.

---

### Scharr Filtering

> OpenCV documentation for image filtering and Scharr derivatives.
>
> 🔗 [OpenCV API](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)

**Usage in this project:** Gradient-based structural representations.

---

## 🔗 Feature Matching & Geometric Verification

> OpenCV feature matching and homography tutorial.
>
> 🔗 [OpenCV Tutorial](https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html)

Implementation-level reference for descriptor matching and homography-based geometric verification.

---

## 🧪 Project-Specific Contributions

The following components are **project-specific implementations** not directly attributed to a single external paper:

| Contribution | Description |
|:--|:--|
| Adaptive orchestration | Fast/recovery path selection |
| Catastrophic-seed gate | Fail-fast rejection of pathological seeds |
| Seed sanity gate | Structured usability assessment |
| 4×4 spatial validation | Grid-based coverage enforcement |
| Area-ratio validation | Transformation scale plausibility |
| Anisotropy validation | Directional distortion check |
| H-seeded optical flow | Homography-initialized LK |
| Balanced point selection | Spatially distributed LK initialization |
| Rotation hypothesis generation | Geometry-driven angle estimation |
| Rotation ambiguity handling | 180° flip testing + local refinement |
| Multi-signal acceptance | Combined final decision gate |

> These mechanisms combine established primitives into the project's **adaptive trust/rejection strategy**.

---

## 🏁 Baselines for Future Benchmarking

The benchmark study may include the following external systems:

| System | Repository |
|:--|:--|
| **SuperGlue** | [magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) |
| **LoFTR** | [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR) |
| **LightGlue** | [cvg/LightGlue](https://github.com/cvg/LightGlue) |
| **Deep Image Matching** | [3DOM-FBK/deep-image-matching](https://github.com/3DOM-FBK/deep-image-matching) |

> [!IMPORTANT]
> These systems should be treated as **benchmark baselines** — not as components of the current hybrid pipeline unless explicitly integrated and evaluated.

---

## 📌 Research Status

> External references establish the underlying methods. Benchmark results, threshold choices, and project-specific validation behavior should be **reported separately** and supported by the project's actual experimental data.
