# X-Palm: Cross-Domain & Cross-Dataset Palmprint Recognition Benchmark

A systematic evaluation framework for palmprint recognition under domain shift, covering cross-dataset generalization, leave-one-out transfer, and within-dataset cross-domain adaptation across four public palmprint datasets.

---

## Table of Contents

1. [Overview](#overview)  
2. [Datasets](#datasets)  
3. [Experiments](#experiments)  
4. [Baselines](#baselines)  
5. [Installation](#installation)  
6. [Data Preparation](#data-preparation)  
7. [Running Experiments](#running-experiments)  
8. [Evaluation Protocol](#evaluation-protocol)  
9. [Results Format](#results-format)  
10. [Citation](#citation)

---

## Overview

Palmprint recognition systems are typically trained and tested on a single dataset under controlled conditions. In real-world deployment, however, models encounter images captured by different devices, under varying lighting, at different distances, and across diverse populations. X-Palm benchmarks how well palmprint models generalize under these realistic domain shifts.

The benchmark consists of **four experiment types**, each probing a different axis of generalization:

| Experiment | Question it answers | # Settings |
|---|---|---|
| Cross-Dataset | Can a model trained on one dataset recognize palms from a completely different dataset? | 16 (4×4 grid) |
| Leave-One-Dataset-Out | Does training on three diverse datasets improve transfer to an unseen fourth? | 4 |
| Cross-Domain Closed-Set | Within one dataset, can a model handle domain shift (e.g., scanner→smartphone) when the same people appear in train and test? | 12 |
| Cross-Domain Open-Set | Same as above, but with completely disjoint identities between train and test (the realistic deployment scenario). | 12 |

All experiments share the same evaluation framework: 512-dimensional normalized embeddings, cosine similarity matching, and reporting of EER (Equal Error Rate) and Rank-1 accuracy.

---

## Datasets

Four palmprint datasets spanning scanner-based, smartphone-based, and multi-device capture:

### CASIA-MS (Multi-Spectral)

A multi-spectral palmprint dataset captured with a dedicated scanner under controlled conditions.

| Property | Value |
|---|---|
| Source | Chinese Academy of Sciences |
| Capture | Dedicated multi-spectral scanner |
| Spectra | Multiple (blue, green, IR, NIR, red, white) |
| Subjects | 200 palms (100 individuals × 2 hands) |
| Images/palm | ~48 (6 spectra × 8 repetitions) |
| Resolution | 128×128 ROI |
| Access | Request from [CASIA Biometrics Ideal](http://biometrics.idealtest.org/) |

**Filename format:** `{subjectID}_{side}_{spectrum}_{repetition}.jpg`
Example: `001_Left_blue_01.jpg`

### Palm-Auth (Smartphone + Scanner)

A dual-modality dataset combining smartphone (perspective) capture under 11 real-world conditions with a multi-spectral desktop scanner.

| Property | Value |
|---|---|
| Capture devices | Smartphone camera + desktop scanner |
| Subjects | 190 palms (perspective), 148 palms (scanner) |
| Perspective conditions | bf, close, far, fl, jf, pitch, rnd, roll, sf, text, wet |
| Scanner spectra | green, ir, yellow, pink, white |
| Resolution | Variable (ROI extracted) |
| Access | [Contact the authors] |

**Directory structure:**
```
smartphone_data/
├── {subject_id}/
│   ├── roi_perspective/
│   │   ├── {id}_{side}_{condition}.jpg         # e.g. 1_left_wet.jpg
│   │   └── {id}_{side}_{condition}_{rep}.jpg   # e.g. 1_left_rnd_1.jpg
│   └── roi_scanner/
│       └── {id}_{Side}_{spectrum}_{rep}.jpg    # e.g. 1_Left_green_01.jpg
```

**Condition descriptions:**

| Condition | Description |
|---|---|
| `rnd` | Random natural pose (rnd_1 through rnd_5) |
| `wet` | Wet/moist palm |
| `text` | Texting posture |
| `sf` | Slight finger spread |
| `roll` | Wrist roll |
| `jf` | Joined fingers |
| `pitch` | Wrist pitch |
| `bf` | Bent fingers |
| `far` | Far distance |
| `close` | Close distance |
| `fl` | Flashlight on |

### MPDv2 (Mobile Palmprint)

A mobile palmprint dataset captured with two smartphone models.

| Property | Value |
|---|---|
| Capture devices | Huawei (h), Motorola (m) |
| Subjects | 400+ palms |
| Sessions | Multiple |
| Resolution | ROI extracted via MediaPipe |
| Access | Contact the dataset authors |

**Filename format:** `{subject}_{session}_{device}_{hand}_{iteration}.jpg`
Example: `001_01_h_l_01.jpg` (Subject 001, session 1, Huawei, left hand, iteration 1)

### XJTU-UP (Xi'an Jiaotong University)

A multi-device palmprint dataset with controlled lighting variations.

| Property | Value |
|---|---|
| Capture devices | iPhone, Huawei |
| Conditions | Flash, Natural light |
| Subjects | 400+ palms |
| Variations | 4 total (2 devices × 2 lighting) |
| Access | [Contact XJTU](http://gr.xjtu.edu.cn/en/web/yihonggong) |

**Directory structure:**
```
XJTU-UP/
├── iPhone/
│   ├── Flash/
│   │   └── {L/R}_{id}/
│   │       └── *.jpg
│   └── Nature/
│       └── {L/R}_{id}/
└── huawei/
    ├── Flash/
    └── Nature/
```

---

## Experiments

### Experiment 1: Cross-Dataset (4×4 Grid)

**Directory:** `cross_dataset/`

Trains a model on one dataset and evaluates on every other dataset (including same-dataset baseline). Tests whether features learned from one palmprint collection transfer to a completely different collection with different demographics, devices, and capture protocols.

**Protocol:**
- **Same-dataset:** 80/20 identity-level split (train on 80% of subjects, test on remaining 20%)
- **Cross-dataset:** Train on ALL subjects from the source, test on ALL subjects from the target
- **Test split:** 50/50 sample-level gallery/probe split within the test set
- **Data balancing:** 190 IDs per dataset (150 "high-sample" IDs × ~29-33 images + 40 "low-sample" IDs × ~14-16 images)

**Output:** 4×4 EER and Rank-1 tables with per-row averages.

### Experiment 2: Leave-One-Dataset-Out

**Directory:** `leave_one_dataset_out/`

Trains on three datasets combined and tests on the fourth. Tests whether training on diverse multi-source data improves generalization compared to single-source training.

**Protocol:**
- **Training:** Combine all samples from 3 datasets (190 IDs each, globally unique label space = 570 classes)
- **Testing:** 50/50 gallery/probe split on the held-out dataset (190 IDs)
- **4 experiments:** Each dataset is left out once

**Output:** Table with one row per held-out dataset + average row.

### Experiment 3: Cross-Domain Closed-Set (Palm-Auth)

**Directory:** `closed_set_cross_domain/`

Evaluates domain shift within Palm-Auth where the **same identities** appear in both training and testing (but from different domains). Tests whether a model can recognize the same person across different capture conditions.

**Settings (12 total):**

| Setting | Train domain | Gallery domain | Probe domain |
|---|---|---|---|
| S_scanner | Perspective (all conditions, 190 IDs) | Scanner (148 IDs, 50%) | Scanner (148 IDs, 50%) |
| S_scanner_to_persp | Scanner (148 IDs) | Perspective (148 IDs, 50%) | Perspective (148 IDs, 50%) |
| S_{A}\_{B} (×10) | Perspective (¬A, ¬B) + Scanner | ALL condition A images | ALL condition B images |

**10 paired conditions:**
(wet, text), (wet, rnd), (rnd, text), (sf, roll), (jf, pitch),
(bf, far), (roll, close), (far, jf), (fl, sf), (roll, pitch)

**Reproducibility:** Gallery/probe splits are saved to `palm_auth_closedset_splits.json` on first run and reused by all baselines.

### Experiment 4: Cross-Domain Open-Set (Palm-Auth)

**Directory:** `open_set_cross_domain/`

Same domain-shift scenarios as Experiment 3, but with **completely disjoint identities** between train and test (80/20 split). This is the realistic deployment scenario where the system encounters people it has never seen during training.

**Protocol:**
- **Identity split:** 80% train (~152 IDs) / 20% test (~38 IDs), no overlap
- **Test evaluation:** 50/50 sample-level gallery/probe within test IDs
- **Same 12 settings** as the closed-set experiment

**Reproducibility:** ID splits are saved to `palm_auth_openset_splits.json`.

---

## Baselines

All baselines share the same evaluation framework (data splits, metrics, gallery/probe construction) and differ only in the model architecture. Each baseline script is self-contained.

### CompNet (Competitive Network with Gabor Filters)

A lightweight model using three parallel Gabor-based competitive blocks at multiple scales, followed by a fully-connected embedding layer and ArcFace classification head.

| Component | Detail |
|---|---|
| Input | 128×128 grayscale |
| Feature extraction | 3 parallel GaborConv2d competitive blocks (kernel sizes 35, 17, 7; strides 3) |
| Embedding | FC: 9708 → 512-d, L2-normalized |
| Loss | CrossEntropy + ArcFace (s=30, m=0.5) |
| Optimizer | Adam (lr=0.001, step decay every 30 epochs, γ=0.8) |
| Augmentation | 2× factor (color jitter, random crop, perspective warp, rotation) |
| Parameters | ~5.2M |

### CCNet

Competitive coding network variant. Same evaluation framework, different backbone architecture.

### DINOv2

Vision transformer backbone (ViT) pretrained with DINOv2 self-supervised learning. Fine-tuned with the same ArcFace head and evaluation protocol.

---

## Installation

```bash
# Create environment
conda create -n xpalm python=3.9 -y
conda activate xpalm

# PyTorch (adjust CUDA version as needed)
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install numpy scipy scikit-learn matplotlib pillow tqdm
```

**Hardware requirements:** Each experiment trains models for 100-300 epochs. A single GPU (RTX 3090 / A100) is sufficient. Multi-GPU is supported via DataParallel.

---

## Data Preparation

### Step 1: Obtain the datasets

| Dataset | How to access |
|---|---|
| CASIA-MS | Register at [CASIA Biometrics Ideal](http://biometrics.idealtest.org/) and request the Multi-Spectral Palmprint dataset. Extract ROIs to a flat directory. |
| Palm-Auth | Contact the authors. The dataset ships with pre-extracted ROIs organized by subject. |
| MPDv2 | Contact the dataset authors. Use MediaPipe-based ROI extraction (or use the provided `MPDv2_mediapipe_manual_roi` version). |
| XJTU-UP | Contact XJTU. The dataset is organized by device/condition/identity. |

### Step 2: Set the data paths

Each experiment script has a `CONFIG` or `BASE_CONFIG` dictionary at the top. Update these paths:

```python
# In each script, update these paths:
"casiams_data_root"   : "/path/to/CASIA-MS-ROI",
"palm_auth_data_root" : "/path/to/smartphone_data",
"mpd_data_root"       : "/path/to/MPDv2_mediapipe_manual_roi",
"xjtu_data_root"      : "/path/to/XJTU-UP",
```

### Step 3: Verify the expected directory structures

**CASIA-MS** (flat directory with all ROI images):
```
CASIA-MS-ROI/
├── 001_Left_blue_01.jpg
├── 001_Left_blue_02.jpg
├── 001_Left_green_01.jpg
└── ...
```

**Palm-Auth** (nested by subject):
```
smartphone_data/
├── 1/
│   ├── roi_perspective/
│   │   ├── 1_left_wet.jpg
│   │   ├── 1_left_rnd_1.jpg
│   │   └── ...
│   └── roi_scanner/
│       ├── 1_Left_green_01.jpg
│       └── ...
├── 2/
└── ...
```

**MPDv2** (flat directory):
```
MPDv2_mediapipe_manual_roi/
├── 001_01_h_l_01.jpg
├── 001_01_h_l_02.jpg
└── ...
```

**XJTU-UP** (nested by device/condition/identity):
```
XJTU-UP/
├── iPhone/
│   ├── Flash/
│   │   ├── L_001/
│   │   │   ├── 001.jpg
│   │   │   └── ...
│   │   └── R_001/
│   └── Nature/
└── huawei/
    ├── Flash/
    └── Nature/
```

---

## Running Experiments

Each experiment is a self-contained Python script. Run from the corresponding directory:

```bash
# Experiment 1: Cross-Dataset (all 16 combinations)
cd cross_dataset
python CompNet_singleScript_allCombinations.py

# Experiment 2: Leave-One-Dataset-Out (4 experiments)
cd leave_one_dataset_out
python CompNet_LeaveOneOut.py

# Experiment 3: Cross-Domain Closed-Set (12 settings)
cd closed_set_cross_domain
python CompNet_CrossDomain_ClosedSet.py

# Experiment 4: Cross-Domain Open-Set (12 settings)
cd open_set_cross_domain
python compnet_crossdomain_openset.py
```

To run a different baseline, replace `CompNet_*` with the corresponding script (e.g., `CCNet_*`, `DINOv2_*`). All baselines produce results in the same format.

### Reproducibility

The code ensures reproducibility through three mechanisms:

1. **Fixed random seed** (`random_seed: 42`) controls all ID splits, data sampling, and gallery/probe assignments.
2. **Cached initial weights:** The first baseline to run saves its randomly initialized weights to disk. All subsequent runs (and all baselines with the same architecture and number of classes) load these weights, ensuring identical starting points.
3. **Persisted splits:** Cross-domain experiments save train/test ID splits and gallery/probe sample assignments to JSON files (`palm_auth_closedset_splits.json`, `palm_auth_openset_splits.json`) on first run. All baselines load the same splits.

---

## Evaluation Protocol

### Embedding Extraction

All models produce 512-dimensional L2-normalized embeddings via a `get_embedding(x)` method that runs the backbone without the classification head or dropout.

### Matching

Cosine similarity between all probe and gallery embeddings produces a full similarity matrix. From this matrix:

**EER (Equal Error Rate):** Every probe-gallery pair is scored. Genuine pairs (same identity) are positive; impostor pairs (different identity) are negative. The EER is the operating point where FAR = FRR, computed via interpolation on the ROC curve.

**Rank-1 Accuracy:** For each probe, the gallery sample with the highest cosine similarity is selected. Rank-1 counts the fraction of probes whose nearest-neighbor gallery match has the correct identity.

### Data Balancing

To ensure fair comparison across datasets with different sizes, each dataset is standardized to 190 identities:

| Group | # Identities | Target samples/ID | Purpose |
|---|---|---|---|
| High | 150 | 29-33 (dataset-dependent) | Sufficient training data |
| Low | 40 | 14-16 (dataset-dependent) | Simulates realistic imbalance |

Sampling is balanced across available spectra/devices/conditions within each identity.

---

## Results Format

Each experiment produces:

```
results_dir/
├── setting_*/                    # One folder per experimental setting
│   ├── eval/
│   │   ├── scores_ep0050_*.txt   # Raw similarity scores + labels
│   │   ├── scores_FINAL_*.txt    # Final evaluation scores
│   │   └── ...
│   ├── net_params.pth            # Final model weights
│   ├── net_params_best_eer.pth   # Best model (lowest EER)
│   ├── train_curves.png          # Loss and accuracy plots
│   └── results.json              # Per-setting EER and Rank-1
├── results_summary.txt           # Formatted results table
├── results_table.txt             # (cross-dataset) Full EER/Rank-1 matrix
└── results_raw.json              # Machine-readable results
```

**Score files** contain one line per probe-gallery pair: `{cosine_similarity} {label}` where label is `1` (genuine) or `-1` (impostor).

---

## Project Structure

```
X-Palm/
├── README.md
├── cross_dataset/
│   ├── CompNet_singleScript_allCombinations.py
│   ├── CCNet_singleScript_allCombinations.py
│   └── DINOv2_singleScript_allCombinations.py
├── leave_one_dataset_out/
│   ├── CompNet_LeaveOneOut.py
│   ├── CCNet_LeaveOneOut.py
│   └── DINOv2_LeaveOneOut.py
├── closed_set_cross_domain/
│   ├── CompNet_CrossDomain_ClosedSet.py
│   ├── CCNet_CrossDomain_ClosedSet.py
│   └── DINOv2_CrossDomain_ClosedSet.py
├── open_set_cross_domain/
│   ├── compnet_crossdomain_openset.py
│   ├── ccnet_crossdomain_openset.py
│   └── dinov2_crossdomain_openset.py
└── palm_auth_*_splits.json       # Auto-generated, shared across baselines
```

---

## Key Hyperparameters

| Parameter | Cross-Dataset | Leave-One-Out | Cross-Domain (Closed) | Cross-Domain (Open) |
|---|---|---|---|---|
| Epochs | 100 | 300 | 200 | 300 |
| Batch size | 128 | 128 | 128 | 128 |
| Learning rate | 0.001 | 0.001 | 0.001 | 0.001 |
| LR schedule | Step (30, γ=0.8) | Step (30, γ=0.8) | Step (30, γ=0.8) | Step (30, γ=0.8) |
| Augmentation | 2× | 2× | 2× | 2× |
| Embedding dim | 512 | 512 | 512 | 512 |
| ArcFace (s, m) | (30, 0.5) | (30, 0.5) | (30, 0.5) | (30, 0.5) |
| Image size | 128×128 | 128×128 | 128×128 | 128×128 |
| Train ID ratio | 0.80 (same-ds) | N/A (full) | N/A (shared IDs) | 0.80 |
| Gallery ratio | 0.50 | 0.50 | 0.50 | 0.50 |
| Eval frequency | Every 50 epochs | Every 50 epochs | Every 50 epochs | Every 50 epochs |

---

## License

[Specify your license here]

## Citation

```bibtex
@article{xpalm2026,
  title   = {X-Palm: A Cross-Domain and Cross-Dataset Benchmark for Palmprint Recognition},
  author  = {},
  journal = {},
  year    = {2026}
}
```
