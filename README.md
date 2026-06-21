# X-Palm: Paired Multispectral-to-Smartphone Dataset for Cross-Domain Palmprint Authentication

**[Paper](https://arxiv.org/abs/2606.08437)** | **[GitHub](https://github.com/X-Palm/X-Palm-2026)** | **[Dataset Access](#dataset-access)**

X-Palm is a cross-domain palmprint dataset comprising **6,006 palm images from 103 individuals (206 hands)**, designed to bridge the gap between controlled multispectral enrollment and unconstrained smartphone authentication. It is the first palmprint dataset providing **paired-identity acquisition** across these two domains, encompassing a broad spectrum of in-the-wild variability.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Benchmarking Code](#benchmarking-code)
4. [Evaluation Protocols](#evaluation-protocols)
5. [Baseline Methods](#baseline-methods)
6. [Installation](#installation)
7. [Data Preparation](#data-preparation)
8. [Running Experiments](#running-experiments)
9. [Implementation Details](#implementation-details)
10. [Results Summary](#results-summary)
11. [Citation](#citation)

---

## Overview

Real-world palmprint authentication requires models to generalize from controlled enrollment to unconstrained mobile verification. Existing datasets capture only partial aspects of this challenge. X-Palm provides paired data for the same identities across two distinct domains:

1. **Controlled Multispectral Enrollment** — captured using a custom-developed multispectral scanner with programmable illumination across six spectral bands (green, blue, red, yellow, white, IR) at 4608×2592 resolution.

2. **Unconstrained Smartphone Authentication** — participant-driven mobile collection using personal devices (80+ models from 10+ brands), deliberately capturing compound variability in hand pose, illumination, distance, perspective, background, occlusion, and palm surface conditions.

Our extensive benchmarks of **12 SOTA models** across three evaluation protocols reveal that existing methods experience severe performance collapse on X-Palm, while models trained on X-Palm demonstrate consistent cross-domain robustness.

---

## Dataset

### X-Palm Statistics

| Property | Value |
|---|---|
| Participants | 103 (61 male, 42 female) |
| Total hands | 206 |
| Total images | 6,006 |
| Paired participants | 81 (both scanner + smartphone) |
| Smartphone-only participants | 22 |
| Scanner images/hand | 18 (3 poses × 6 spectra) |
| Smartphone images/hand | 15 (10 structured + 5 random) |
| ROI size | 112 × 112 |
| Age range | 18–76 |
| Ethnic groups | 11 self-reported |
| Smartphone brands | 10+ (80+ distinct models) |

### Acquisition Domains

**Multispectral Scanner (Controlled):**
Six spectral bands — Green, Blue, Red, Yellow, White, and IR — captured across three hand poses (separated fingers, bent fingers, joined fingers).

**Smartphone (Unconstrained):**
Ten structured variations plus five random captures per hand:

| Variation | Abbreviation | Description |
|---|---|---|
| Bent Fingers | BF | Fingers curled inward |
| Joined Fingers | JF | Fingers pressed together |
| Separated Fingers | SF | Fingers spread apart |
| Close | Close | Hand held close to camera |
| Far | Far | Hand held far from camera |
| Flash ON | FO | Smartphone flash activated |
| Pitch | Pitch | Wrist tilted forward/backward |
| Roll | Roll | Wrist rotated sideways |
| Text | Text | Handwritten text on palm surface |
| Wet | Wet | Moisture on palm surface |
| Random 1–5 | RND | Arbitrary combinations of the above |

### Additional Benchmark Datasets

Cross-dataset evaluation uses three additional public datasets:

| Dataset | #Hands | #Images | Devices | Conditions |
|---|---|---|---|---|
| CASIA-MS | 200 | 7,200 | 1 scanner | 6 spectra, controlled |
| MPD-v2 | 400 | 16,000 | 2 smartphones | Background + lighting variation |
| XJTU-UP | 200 | 20,000 | 5 smartphones | 2 lighting conditions |

### Dataset Access

X-Palm is released for **non-commercial academic use only**. To obtain access:

1. Review the End User License Agreement (EULA) available in the [repository](https://github.com/X-Palm/X-Palm-2026)
2. Submit a signed EULA as instructed on the repository page
3. Approved requests receive a time-limited secure download link within 24 hours

The EULA prohibits redistribution, re-identification attempts, and commercial exploitation.

For the public benchmark datasets:
- **CASIA-MS:** Request from [CASIA Biometrics Ideal](http://biometrics.idealtest.org/)
- **MPD-v2:** Contact the dataset authors ([Zhang et al., 2020](https://arxiv.org/abs/2003.13266))
- **XJTU-UP:** Contact [XJTU](http://gr.xjtu.edu.cn/en/web/yihonggong) ([Shao et al., 2019](https://openaccess.thecvf.com/content_CVPRW_2019/html/Biometrics/Shao_Efficient_Deep_Palmprint_Recognition_via_Distilled_Hashing_Coding_CVPRW_2019_paper.html))

---

## Benchmarking Code

The repository provides self-contained experiment scripts organized by evaluation protocol:

```
X-Palm-2026/
├── README.md
├── cross_dataset/
│   ├── CompNet_singleScript_allCombinations.py
│   ├── CCNet_singleScript_allCombinations.py
│   ├── CO3Net_singleScript_allCombinations.py
│   ├── PPNet_singleScript_allCombinations.py
│   ├── SF2Net_singleScript_allCombinations.py
│   ├── PalmBridge_singleScript_allCombinations.py
│   ├── ConvNeXtV2_singleScript_allCombinations.py
│   ├── DINOv2_singleScript_allCombinations.py
│   ├── ArcFace_singleScript_allCombinations.py
│   └── MagFace_singleScript_allCombinations.py
├── closed_set_cross_domain/
│   ├── CompNet_CrossDomain_ClosedSet.py
│   ├── ... (one per baseline)
│   └── palm_auth_closedset_splits.json      ← auto-generated
├── open_set_cross_domain/
│   ├── CompNet_CrossDomain_OpenSet.py
│   ├── ... (one per baseline)
│   └── palm_auth_openset_splits.json        ← auto-generated
└── leave_one_dataset_out/
    ├── CompNet_LeaveOneOut.py
    └── ... (one per baseline)
```

Each script is **fully self-contained**: it includes the model definition, dataset parsing, training loop, evaluation logic, and results reporting. No external model library is required beyond PyTorch and timm.

---

## Evaluation Protocols

### Protocol 1: Cross-Dataset (4×4 Grid)

Models are trained on one dataset and evaluated on every other dataset. Tests whether features generalize across completely different collections, demographics, and capture devices.

- **Train/Test datasets:** X-Palm, CASIA-MS, MPD-v2, XJTU-UP
- **Same-dataset:** 80/20 identity-level split
- **Cross-dataset:** Train on all source identities, test on all target identities
- **Test split:** 50/50 sample-level gallery/probe
- **Result:** 4×4 matrix of EER and Rank-1 with per-row averages

### Protocol 2: Closed-Set Cross-Domain (within X-Palm)

Models are trained on specific acquisition domains and tested on held-out domains. **Identities are shared** between train and test (Y_train = Y_test), isolating domain shift as the sole difficulty.

**12 settings:**

| Setting | Train Domain | Test Domain (Gallery / Probe) |
|---|---|---|
| Scanner | All smartphone domains | Multispectral scanner (50/50 split) |
| Smartphone | Multispectral scanner | All smartphone domains (50/50 split) |
| Wet & Text | All except Wet, Text + Scanner | Gallery: Wet / Probe: Text |
| Wet & RND | All except Wet, RND + Scanner | Gallery: Wet / Probe: RND |
| RND & Text | All except RND, Text + Scanner | Gallery: RND / Probe: Text |
| SF & Roll | All except SF, Roll + Scanner | Gallery: SF / Probe: Roll |
| JF & Pitch | All except JF, Pitch + Scanner | Gallery: JF / Probe: Pitch |
| BF & Far | All except BF, Far + Scanner | Gallery: BF / Probe: Far |
| Roll & Close | All except Roll, Close + Scanner | Gallery: Roll / Probe: Close |
| Far & JF | All except Far, JF + Scanner | Gallery: Far / Probe: JF |
| FO & SF | All except FO, SF + Scanner | Gallery: FO / Probe: SF |
| Roll & Pitch | All except Roll, Pitch + Scanner | Gallery: Roll / Probe: Pitch |

### Protocol 3: Open-Set Cross-Domain (within X-Palm)

Same domain-shift scenarios as Protocol 2, but with **completely disjoint identities** (Y_train ∩ Y_test = ∅). This is the realistic deployment scenario.

- **Identity split:** 80% train / 20% test, no overlap
- **Test evaluation:** 50/50 sample-level gallery/probe within test identities
- **Same 12 settings** as Protocol 2

### Protocol 4: Leave-One-Dataset-Out (Appendix)

Models are trained on K−1 datasets combined and evaluated on the held-out dataset. Tests whether multi-source training improves cross-dataset transfer.

- **4 experiments:** Each of the four datasets is held out once
- **Result:** Table with EER and Rank-1 per held-out dataset + average

---

## Baseline Methods

12 SOTA models spanning palmprint-specific architectures, domain adaptation methods, pretrained vision models, and face-pretrained models:

### Palmprint-Specific Models

| Model | Params | GFLOPs | Optimizer | Key Feature |
|---|---|---|---|---|
| CompNet | 3.27M | 0.735 | Adam (lr=1e-3) | Learnable Gabor kernels, competitive blocks |
| PPNet | 3.53M | 0.735 | Adam (lr=1e-3) | Dual-camera alignment |
| CCNet | 20.57M | 2.131 | Adam (lr=1e-3) | Comprehensive competition mechanism |
| CO3Net | 20.57M | 2.131 | Adam (lr=1e-3) | Coordinate-aware contrastive competitive |
| SF2Net | 13.08M | 2.655 | Adam (lr=1e-3) | Sequence feature fusion |
| PalmBridge | 3.53M | 0.735 | Adam (lr=1e-3) | Plug-and-play feature alignment (CompNet backbone) |

### Domain Adaptation / Generalization

| Model | Params | GFLOPs | Key Feature |
|---|---|---|---|
| TSCAN | 11.31M | 0.486 | Teacher-student co-learning with EMA (decay=0.999) |
| GIFT | 11.24M | 0.486 | Stylized feature generation for single-source DG |

### Pretrained Vision Models

| Model | Params | GFLOPs | Optimizer | Frozen Layers |
|---|---|---|---|---|
| ConvNeXtV2-Tiny | 27.87M | 1.067 | AdamW (lr=1e-3, cosine) | All except last stage |
| DINOv2 ViT-S/14 | 22.06M | 1.398 | AdamW (lr=1e-3, cosine) | All except last 2 blocks |

### Face-Pretrained Models

| Model | Params | GFLOPs | Optimizer | Pretraining |
|---|---|---|---|---|
| ArcFace-iResNet100 | 65.12M | 12.098 | AdamW (lr=1e-4, wd=5e-4, cosine) | Glint360K |
| MagFace-iResNet100 | 65.16M | 12.117 | AdamW (lr=1e-4, wd=5e-4, cosine) | MS1MV2 |

All models output **512-d L2-normalized embeddings** and use **ArcFace** loss for training (except MagFace which uses its adaptive margin formulation).

---

## Installation

```bash
conda create -n xpalm python=3.9 -y && conda activate xpalm

# PyTorch (adjust CUDA version as needed)
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install timm numpy scipy scikit-learn matplotlib pillow tqdm
```

**Hardware:** All experiments were conducted on an NVIDIA RTX A6000 (48 GB). Single-GPU training is sufficient; multi-GPU is supported via DataParallel.

---

## Data Preparation

### Step 1: Obtain datasets

Follow the access instructions in the [Dataset Access](#dataset-access) section above for each dataset.

### Step 2: Set data paths

Each experiment script contains a `CONFIG` (or `BASE_CONFIG`) dictionary at the top. Update the dataset root paths:

```python
BASE_CONFIG = {
    "palm_auth_data_root"  : "/path/to/X-Palm",          # X-Palm dataset
    "casiams_data_root"    : "/path/to/CASIA-MS-ROI",    # CASIA-MS ROI images
    "mpd_data_root"        : "/path/to/MPDv2_ROI",       # MPD-v2 ROI images
    "xjtu_data_root"       : "/path/to/XJTU-UP",         # XJTU-UP dataset
    ...
}
```

### Step 3: Verify directory structures

**X-Palm** (nested by subject):
```
X-Palm/
├── {subject_id}/
│   ├── roi_perspective/
│   │   ├── {id}_{side}_{condition}.jpg         # e.g. 1_left_wet.jpg
│   │   └── {id}_{side}_{condition}_{rep}.jpg   # e.g. 1_left_rnd_1.jpg
│   └── roi_scanner/
│       └── {id}_{Side}_{spectrum}_{rep}.jpg    # e.g. 1_Left_green_01.jpg
├── {subject_id}/
└── ...
```

**CASIA-MS** (flat directory):
```
CASIA-MS-ROI/
├── {id}_{side}_{spectrum}_{rep}.jpg
└── ...
```

**MPD-v2** (flat directory):
```
MPDv2_ROI/
├── {subject}_{session}_{device}_{hand}_{iteration}.jpg
└── ...
```

**XJTU-UP** (nested by device/condition/identity):
```
XJTU-UP/
├── iPhone/
│   ├── Flash/{L,R}_{id}/*.jpg
│   └── Nature/{L,R}_{id}/*.jpg
└── huawei/
    ├── Flash/{L,R}_{id}/*.jpg
    └── Nature/{L,R}_{id}/*.jpg
```

---

## Running Experiments

Each experiment is a self-contained Python script:

```bash
# Protocol 1: Cross-Dataset (4×4 grid, all 12 baselines)
cd cross_dataset
python CompNet_singleScript_allCombinations.py

# Protocol 2: Closed-Set Cross-Domain (12 settings)
cd closed_set_cross_domain
python CompNet_CrossDomain_ClosedSet.py

# Protocol 3: Open-Set Cross-Domain (12 settings)
cd open_set_cross_domain
python CompNet_CrossDomain_OpenSet.py

# Protocol 4: Leave-One-Dataset-Out (4 experiments)
cd leave_one_dataset_out
python CompNet_LeaveOneOut.py
```

Replace `CompNet` with any other baseline name to run that model under the same evaluation framework.

---

## Implementation Details

All details follow Appendix A.2 of the paper:

| Parameter | Palmprint Models | ConvNeXtV2 / DINOv2 | ArcFace / MagFace |
|---|---|---|---|
| Input size | 112 × 112 | 112 × 112 | 112 × 112 |
| Epochs | 200 | 200 | 200 |
| Batch size | 64 | 64 | 64 |
| Optimizer | Adam | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-3 | 1e-4 |
| LR schedule | Step decay | Cosine annealing | Cosine annealing |
| Weight decay | — | — | 5e-4 |
| Frozen layers | None | All except last stage/2 blocks | First 75% of tensors |
| Runs | 3 (independent seeds) | 3 | 3 |

**Augmentation** (shared across all models): contrast jitter, random crop, perspective distortion, and edge-anchored rotation.

**Matching:** Cosine similarity on L2-normalized embeddings.

**Reproducibility safeguards:**
1. Gallery/probe splits generated once with a fixed seed and shared across all models (saved to JSON).
2. Initial weights cached after first run and reused identically across all train-test combinations.
3. All results are mean values across 3 independent runs.

---

## Results Summary

### Cross-Dataset: Key Finding

Models trained on X-Palm achieve the **best average cross-dataset generalization**, demonstrating that exposure to compound in-the-wild variations forces models to learn more robust features. Conversely, models trained on other datasets experience severe performance degradation when tested on X-Palm (EER >33% for all models).

### Cross-Domain: Key Finding

Cross-setting authentication (Scanner↔Smartphone) is extremely challenging for all baselines (average EER >25%). Among in-the-wild conditions, **Roll, Pitch, Wet, Text, and Random** present the greatest difficulty due to severe intra-class variability from pose distortion, surface occlusion, and unpredictable compound variations.

### Leave-One-Out: Key Finding

X-Palm is the most challenging held-out test set (average EER=35.05%), confirming that simply scaling multi-source training data cannot overcome cross-setting and cross-domain challenges without methods designed for compound variability.

---

## Results Format

Each experiment produces:

```
results_dir/
├── setting_*/
│   ├── eval/
│   │   └── scores_{tag}.txt    # Per-pair: {cosine_similarity} {1 or -1}
│   ├── net_params_best_eer.pth # Best model checkpoint
│   ├── train_curves.png        # Loss and accuracy plots
│   └── results.json            # Per-setting EER and Rank-1
├── results_summary.txt         # Formatted results table
└── results_raw.json            # Machine-readable results
```

---

## Citation

```bibtex
@article{seyedmohammadi2026xpalm,
  title     = {X-Palm: Paired Multispectral-to-Smartphone Dataset for Cross-Domain Palmprint Authentication},
  author    = {Seyedmohammadi, Jamal and Ng, Pai Chet and Genovese, Angelo and Chi, Zhixiang and Lee, Jeannie and Plataniotis, Konstantinos N.},
  journal   = {Preprint},
  year      = {2026},
  url       = {https://github.com/X-Palm/X-Palm-2026}
}
```

## License

X-Palm is released under a strict non-commercial End User License Agreement (EULA). Usage is restricted to academic research. See the EULA in the repository for full terms.
