"""
palmbridge_openset.py — PalmBridge: Cross-Domain Open-Set Evaluations on X-Palm
=====================================================================================
Model, training method, and hyperparameters: unchanged from original PalmBridge.
Dataset      : X-Palm (roi_perspective + roi_scanner)
Evaluation   : Cross-Domain Open-Set (12 settings)

Open-set protocol
──────────────────
  Subject IDs are split 80 % / 20 % into DISJOINT train and test partitions.
  train_ratio = 0.80  
  test_ratio  = 0.20  

  Gallery and probe are built from TEST IDs only, with a 50 / 50
  sample-level split so every test identity appears in both sets.

Settings (12 total)
────────────────────
  S_scanner         │ Train : perspective (all)   for train IDs
                    │ Test  : scanner             for test IDs

  S_scanner_to_persp│ Train : scanner             for scanner IDs
                    │ Test  : perspective (all)   for IDs with NO scanner data

  S_(A,B) (×10)     │ Train : perspective(¬A,¬B) + scanner  for train IDs
                    │ Test  : 1 img from A → gallery / 1 img from B → probe
                    │         (random assignment), test IDs only

Gallery/probe splits are saved to x_palm_openset_splits.json on first run
and reused by all models for fair cross-model comparison.

Results saved to:
  {SAVE_DIR}/setting_scanner/
  {SAVE_DIR}/setting_{A}_{B}/
  {SAVE_DIR}/results_summary.txt
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          PARAMETERS                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Data ─────────────────────────────────────────────────────────────────────
X_PALM_ROOT  = "path to x-palm data"
SCANNER_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

# ── Open-set evaluation split ─────────────────────────────────────────────────
TRAIN_ID_RATIO     = 0.80   # fraction of IDs used for training
TEST_GALLERY_RATIO = 0.50   # sample-level gallery/probe split within test IDs
SPLITS_FILE        = "./x_palm_openset_splits.json"

# ── Feature blending ──────────────────────────────────────────────────────────
W_MAP = 0.3
W_ORI = 0.7

# ── Plug-and-play mode ────────────────────────────────────────────────────────
PLUG_AND_PLAY = False
PB_CKPT       = None

# ── Model architecture ────────────────────────────────────────────────────────
FEATURE_DIM       = 512
NUM_PB_VECTORS    = 512
NUM_GABOR_FILTERS = 32
GABOR_KERNEL_SIZE = 15

# ── Loss weights ──────────────────────────────────────────────────────────────
ALPHA      = 0.1
BETA       = 1.0
LAMBDA_CON = 0.25

# ── ArcFace ───────────────────────────────────────────────────────────────────
ARC_S = 48.0
ARC_M = 0.40

# ── Training ──────────────────────────────────────────────────────────────────
IMG_SIZE     = 128
BATCH_SIZE   = 16
LR           = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS       = 100
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS  = 4

# ── PalmBridge warm-up ────────────────────────────────────────────────────────
WARMUP_EPOCHS = 5

# ── Misc ──────────────────────────────────────────────────────────────────────
N_EER_THRESHOLDS = 2000
SEED             = 42
SAVE_DIR         = "./rst_palmbridge_crossdomain_openset"
PLOT_DIR         = "./rst_palmbridge_crossdomain_openset/plots"
LOG_EPOCHS       = 5

# ── Paired conditions ─────────────────────────────────────────────────────────
PAIRED_CONDITIONS = [
    ("wet",  "text"),
    ("wet",  "rnd"),
    ("rnd",  "text"),
    ("sf",   "roll"),
    ("jf",   "pitch"),
    ("bf",   "far"),
    ("roll", "close"),
    ("far",  "jf"),
    ("fl",   "sf"),
    ("roll", "pitch"),
]

# ╚═══════════════════════════════════════════════════════════════════════════╝

IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ══════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════
#  MODEL  (unchanged from original PalmBridge)
# ══════════════════════════════════════════════════════════════

class LearnableGaborLayer(nn.Module):
    def __init__(self):
        super().__init__()
        n  = NUM_GABOR_FILTERS
        ks = GABOR_KERNEL_SIZE
        self.theta = nn.Parameter(torch.linspace(0.0, math.pi, n + 1)[:-1])
        self.sigma = nn.Parameter(torch.full((n,), 3.0))
        self.lambd = nn.Parameter(torch.full((n,), 6.0))
        self.psi   = nn.Parameter(torch.zeros(n))
        self.gamma = nn.Parameter(torch.full((n,), 0.5))
        half = ks // 2
        ys   = torch.arange(-half, half + 1, dtype=torch.float32)
        xs   = torch.arange(-half, half + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)
        self.ks = ks

    def _build_filters(self) -> torch.Tensor:
        theta = self.theta
        sigma = self.sigma.abs().clamp(min=0.5)
        lambd = self.lambd.abs().clamp(min=1.0)
        psi   = self.psi
        gamma = self.gamma.abs().clamp(min=0.1)
        xx = self.xx.unsqueeze(0)
        yy = self.yy.unsqueeze(0)
        cos_t = torch.cos(theta).view(-1, 1, 1)
        sin_t = torch.sin(theta).view(-1, 1, 1)
        sigma = sigma.view(-1, 1, 1)
        lambd = lambd.view(-1, 1, 1)
        psi   = psi.view(-1, 1, 1)
        gamma = gamma.view(-1, 1, 1)
        x_rot =  xx * cos_t + yy * sin_t
        y_rot = -xx * sin_t + yy * cos_t
        envelope = torch.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2.0 * sigma**2))
        kernel   = envelope * torch.cos(2.0 * math.pi * x_rot / lambd + psi)
        kernel   = kernel - kernel.mean(dim=(1, 2), keepdim=True)
        return kernel.unsqueeze(1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filters = self._build_filters()
        return F.conv2d(x, filters, padding=self.ks // 2)


class CompetitivePool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = x.abs().max(dim=1, keepdim=True)
        return out


class CompNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.gabor   = LearnableGaborLayer()
        self.compete = CompetitivePool()
        self.gbn     = nn.BatchNorm2d(1)

        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin,  cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.block1 = _block(1,   32);  self.pool1 = nn.MaxPool2d(2)
        self.block2 = _block(32,  64);  self.pool2 = nn.MaxPool2d(2)
        self.block3 = _block(64,  128); self.pool3 = nn.MaxPool2d(2)
        self.block4 = _block(128, 256); self.gap   = nn.AdaptiveAvgPool2d((4, 4))
        self.embed  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, FEATURE_DIM, bias=False),
            nn.BatchNorm1d(FEATURE_DIM),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.relu(self.gabor(x))
        g = self.gbn(self.compete(g))
        f = self.pool1(self.block1(g))
        f = self.pool2(self.block2(f))
        f = self.pool3(self.block3(f))
        f = self.gap(self.block4(f))
        z = self.embed(f)
        return F.normalize(z, p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  PALMBRIDGE MODULE  (unchanged from original)
# ══════════════════════════════════════════════════════════════

class PalmBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = NUM_PB_VECTORS
        self.P = nn.Parameter(
            F.normalize(torch.randn(NUM_PB_VECTORS, FEATURE_DIM), p=2, dim=1))

    def _nearest_vector(self, z):
        dists = (z.pow(2).sum(1, keepdim=True)
                 + self.P.pow(2).sum(1).unsqueeze(0)
                 - 2.0 * (z @ self.P.t()))
        idx     = dists.argmin(dim=1)
        z_tilde = self.P[idx]
        return z_tilde, idx

    def _blend(self, z, z_tilde):
        return W_ORI * z + W_MAP * z_tilde

    def forward(self, z):
        z_tilde, indices = self._nearest_vector(z)
        z_hat            = self._blend(z, z_tilde)
        return z_hat, z_tilde, indices

    def loss_consistency(self, z, z_tilde):
        t1 = (z_tilde - z.detach()).pow(2).sum(1).mean()
        t2 = (z - z_tilde.detach()).pow(2).sum(1).mean()
        return t1 + LAMBDA_CON * t2

    def loss_orthogonal(self):
        W = F.normalize(self.P, p=2, dim=1)
        S = W @ W.t()
        I = torch.eye(self.K, device=S.device, dtype=S.dtype)
        return ((S - I).pow(2)).sum() / (self.K ** 2)

    @torch.no_grad()
    def codebook_usage(self):
        W   = F.normalize(self.P, p=2, dim=1)
        S   = W @ W.t()
        off = S[~torch.eye(self.K, dtype=torch.bool, device=S.device)]
        return {"mean_cosine":    off.mean().item(),
                "near_duplicate": (off > 0.9).float().mean().item()}


# ══════════════════════════════════════════════════════════════
#  ARCFACE LOSS  (unchanged from original)
# ══════════════════════════════════════════════════════════════

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.s     = ARC_S
        self.cos_m = math.cos(ARC_M)
        self.sin_m = math.sin(ARC_M)
        self.th    = math.cos(math.pi - ARC_M)
        self.mm    = math.sin(math.pi - ARC_M) * ARC_M
        self.weight = nn.Parameter(torch.empty(num_classes, FEATURE_DIM))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z, labels):
        W           = F.normalize(self.weight, p=2, dim=1)
        cos_theta   = (z @ W.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta   = (1.0 - cos_theta**2).sqrt()
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m,
                                  cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1.0)
        logits  = self.s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)
        return self.ce(logits, labels)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION  (Palm-Auth style)
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    def __init__(self, outchannels=1): self.outchannels = outchannels
    def __call__(self, tensor):
        c, h, w = tensor.size(); tensor = tensor.view(c, h * w)
        idx = tensor > 0; t = tensor[idx]
        if t.numel() > 1:
            tensor[idx] = t.sub_(t.mean()).div_(t.std() + 1e-6)
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATASETS
# ══════════════════════════════════════════════════════════════

def get_train_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.15, p=1),
            transforms.RandomChoice([
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(0.5*IMG_SIZE, 0.0)),
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(0.0, 0.5*IMG_SIZE)),
            ]),
        ]),
        transforms.ToTensor(), NormSingleROI(outchannels=1),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        NormSingleROI(outchannels=1),
    ])


class PalmAuthDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], train: bool = False):
        self.samples   = samples
        self.transform = get_train_transform() if train else get_eval_transform()

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label


def make_loader(samples, train=False, shuffle=False):
    ds = PalmAuthDataset(samples, train=train)
    return DataLoader(ds, batch_size=min(BATCH_SIZE, len(samples)),
                      shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=(DEVICE == "cuda"),
                      drop_last=train and len(samples) > BATCH_SIZE)


# ══════════════════════════════════════════════════════════════
#  DATA COLLECTION HELPERS
# ══════════════════════════════════════════════════════════════

def _collect_perspective(data_root):
    """condition → identity → [path, ...]"""
    cond_paths = defaultdict(lambda: defaultdict(list))
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        roi_dir = os.path.join(subject_dir, "roi_perspective")
        if not os.path.isdir(roi_dir): continue
        for fname in sorted(os.listdir(roi_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 3: continue
            identity  = parts[0] + "_" + parts[1].lower()
            condition = parts[2].lower()
            cond_paths[condition][identity].append(os.path.join(roi_dir, fname))
    return cond_paths


def _collect_scanner(data_root, scanner_spectra):
    """identity → [path, ...]"""
    scanner_paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        scan_dir = os.path.join(subject_dir, "roi_scanner")
        if not os.path.isdir(scan_dir): continue
        for fname in sorted(os.listdir(scan_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 4: continue
            if parts[2].lower() not in scanner_spectra: continue
            identity = parts[0] + "_" + parts[1].lower()
            scanner_paths[identity].append(os.path.join(scan_dir, fname))
    return scanner_paths


def _all_samples(id2paths, label_map):
    return [(p, label_map[ident])
            for ident, paths in id2paths.items()
            for p in paths]


# ══════════════════════════════════════════════════════════════
#  OPEN-SET TRAIN/TEST ID SPLIT PERSISTENCE
# ══════════════════════════════════════════════════════════════

def generate_all_splits(cond_paths, scanner_paths, train_id_ratio, seed):
    """
    Determine disjoint train/test ID splits for all 12 settings.
    Called once and saved to SPLITS_FILE so every model uses identical splits.
    """
    _rng = random.Random(seed)

    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    all_persp_ids = sorted(persp_all.keys())   # 190
    scanner_ids   = sorted(scanner_paths.keys())  # 148
    n_test = len(all_persp_ids) - int(len(all_persp_ids) * train_id_ratio)  # ~38

    splits = {}

    # S_scanner: test IDs sampled from scanner IDs (they have both modalities),
    #            train IDs = remaining perspective IDs
    test_ids  = sorted(_rng.sample(scanner_ids, min(n_test, len(scanner_ids))))
    train_ids = sorted(set(all_persp_ids) - set(test_ids))
    splits["S_scanner"] = {"train_ids": train_ids, "test_ids": test_ids}

    # S_scanner_to_persp: train = all scanner IDs,
    #                     test  = perspective-only IDs (no scanner data)
    no_scanner_ids = sorted(set(all_persp_ids) - set(scanner_ids))
    splits["S_scanner_to_persp"] = {
        "train_ids": scanner_ids,
        "test_ids" : no_scanner_ids,
    }

    # Paired-condition settings
    for cond_a, cond_b in PAIRED_CONDITIONS:
        paths_a      = cond_paths.get(cond_a, {})
        paths_b      = cond_paths.get(cond_b, {})
        eligible_ids = sorted(set(paths_a.keys()) & set(paths_b.keys()))
        if not eligible_ids:
            print(f"  [WARN] No IDs with both '{cond_a}' and '{cond_b}' — skipping split")
            continue
        n_t       = min(n_test, len(eligible_ids))
        test_ids  = sorted(_rng.sample(eligible_ids, n_t))
        train_ids = sorted(set(all_persp_ids) - set(test_ids))
        splits[f"S_{cond_a}_{cond_b}"] = {
            "train_ids": train_ids,
            "test_ids" : test_ids,
        }

    return splits


def load_or_generate_splits(cond_paths, scanner_paths, train_id_ratio, seed):
    """
    Load splits from SPLITS_FILE if it exists; otherwise generate, save, and return.
    """
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE) as f:
            splits = json.load(f)
        print(f"  Loaded existing ID splits from: {SPLITS_FILE}")
    else:
        print(f"  Generating ID splits (seed={seed}) → {SPLITS_FILE}")
        splits = generate_all_splits(cond_paths, scanner_paths, train_id_ratio, seed)
        with open(SPLITS_FILE, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"  Splits saved to: {SPLITS_FILE}")

    for key, val in splits.items():
        print(f"    {key:<30}  train={len(val['train_ids'])}  test={len(val['test_ids'])}")
    return splits


# ══════════════════════════════════════════════════════════════
#  GALLERY/PROBE SAMPLE-LEVEL SPLIT (within test IDs)
# ══════════════════════════════════════════════════════════════

def _gallery_probe_split(id2paths, label_map, gallery_ratio, rng):
    """50/50 sample-level split — every test identity appears in both sets."""
    gallery, probe = [], []
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        if len(paths) == 1:
            # Edge case: duplicate single image into both sets
            gallery.append((paths[0], label_map[ident]))
            probe.append((paths[0], label_map[ident]))
        else:
            n_gal = min(n_gal, len(paths) - 1)   # guarantee ≥1 probe
            for p in paths[:n_gal]: gallery.append((p, label_map[ident]))
            for p in paths[n_gal:]: probe.append((p, label_map[ident]))
    return gallery, probe


# ══════════════════════════════════════════════════════════════
#  PARSERS FOR EACH OPEN-SET SETTING
# ══════════════════════════════════════════════════════════════

def parse_setting_scanner(cond_paths, scanner_paths, splits, seed):
    """
    S_scanner — open-set.
    Train (80 %): ALL perspective images for train IDs.
    Test  (20 %): scanner images for test IDs → 50/50 gallery/probe.
    """
    rng = random.Random(seed)

    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    train_ids = splits["train_ids"]
    test_ids  = splits["test_ids"]

    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(test_ids)}
    num_train_cls   = len(train_ids)

    train_samples = _all_samples(
        {i: persp_all[i] for i in train_ids if i in persp_all},
        train_label_map)

    gallery_samples, probe_samples = _gallery_probe_split(
        {i: scanner_paths[i] for i in test_ids if i in scanner_paths},
        test_label_map, TEST_GALLERY_RATIO, rng)

    _print_stats(
        "S_scanner | Perspective (train IDs) → Scanner (test IDs)",
        len(train_ids), len(test_ids),
        len(train_samples), len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_scanner_to_perspective(cond_paths, scanner_paths, splits, seed):
    """
    S_scanner_to_persp — open-set.
    Train: ALL scanner images for scanner IDs.
    Test : ALL perspective images for no-scanner IDs → 50/50 gallery/probe.
    """
    rng = random.Random(seed)

    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    train_ids = splits["train_ids"]   # scanner IDs
    test_ids  = splits["test_ids"]    # no-scanner perspective IDs

    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(test_ids)}
    num_train_cls   = len(train_ids)

    train_samples = _all_samples(scanner_paths, train_label_map)

    gallery_samples, probe_samples = _gallery_probe_split(
        {i: persp_all[i] for i in test_ids if i in persp_all},
        test_label_map, TEST_GALLERY_RATIO, rng)

    _print_stats(
        "S_scanner_to_persp | Scanner (all) → Perspective (no-scanner IDs)",
        len(train_ids), len(test_ids),
        len(train_samples), len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_paired_conditions(cond_a, cond_b, cond_paths, scanner_paths,
                                    splits, seed):
    """
    S_{A}_{B} — open-set.
    Train: perspective (all except cond_A and cond_B) + scanner, train IDs only.
    Test : 1 img from cond_A and 1 img from cond_B per test identity;
           random coin-flip assigns which condition goes to gallery vs probe.
    """
    rng = random.Random(seed)

    paths_a = cond_paths.get(cond_a, {})
    paths_b = cond_paths.get(cond_b, {})
    if not paths_a: raise ValueError(f"No images for condition '{cond_a}'")
    if not paths_b: raise ValueError(f"No images for condition '{cond_b}'")

    train_ids = splits["train_ids"]
    test_ids  = splits["test_ids"]

    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(test_ids)}
    num_train_cls   = len(train_ids)

    # Train: all perspective except both test conditions + scanner, train IDs only
    train_samples = []
    for cond, cond_dict in cond_paths.items():
        if cond in (cond_a, cond_b):
            continue
        for ident in train_ids:
            for p in cond_dict.get(ident, []):
                train_samples.append((p, train_label_map[ident]))
    for ident in train_ids:
        for p in scanner_paths.get(ident, []):
            train_samples.append((p, train_label_map[ident]))

    # Test: 1 sample from cond_A, 1 from cond_B per test identity
    # Random coin-flip decides which condition goes to gallery
    gallery_samples, probe_samples = [], []
    for ident in test_ids:
        label  = test_label_map[ident]
        a_imgs = list(paths_a.get(ident, [])); rng.shuffle(a_imgs)
        b_imgs = list(paths_b.get(ident, [])); rng.shuffle(b_imgs)
        if not a_imgs or not b_imgs:
            continue
        if rng.random() < 0.5:
            gallery_samples.append((a_imgs[0], label))
            probe_samples.append((b_imgs[0], label))
        else:
            gallery_samples.append((b_imgs[0], label))
            probe_samples.append((a_imgs[0], label))

    _print_stats(
        f"S_{cond_a}_{cond_b} | Perspective(not {cond_a}/{cond_b})+Scanner"
        f" → gallery:{cond_a}/{cond_b} (random) / probe:{cond_b}/{cond_a}",
        len(train_ids), len(test_ids),
        len(train_samples), len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def _print_stats(name, n_train_ids, n_test_ids, train_n, gallery_n, probe_n):
    print(f"\n  [{name}]")
    print(f"    Train IDs / Test IDs  : {n_train_ids} / {n_test_ids}")
    print(f"    Train images          : {train_n}")
    print(f"    Gallery / Probe       : {gallery_n} / {probe_n}")


# ══════════════════════════════════════════════════════════════
#  TRAINING  (unchanged from original PalmBridge)
# ══════════════════════════════════════════════════════════════

def train_one_epoch(backbone, palmbridge, arcface, loader, optimizer, epoch):
    backbone.train(); palmbridge.train(); arcface.train()
    totals    = {"loss": 0.0, "bak": 0.0, "con": 0.0, "orth": 0.0}
    n_correct = 0; n_total = 0
    pb_active = (epoch > WARMUP_EPOCHS)

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        z                 = backbone(imgs)
        z_hat, z_tilde, _ = palmbridge(z)
        feat_for_arc      = z_hat if pb_active else z
        L_bak = arcface(feat_for_arc, labels)
        L_con = palmbridge.loss_consistency(z, z_tilde)
        L_o   = palmbridge.loss_orthogonal()
        loss  = L_bak + ALPHA * L_con + BETA * L_o
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(backbone.parameters()) + list(palmbridge.parameters()),
            max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            W      = F.normalize(arcface.weight, p=2, dim=1)
            logits = ARC_S * (z_hat @ W.t())
            preds  = logits.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total   += labels.size(0)
        totals["loss"] += loss.item()
        totals["bak"]  += L_bak.item()
        totals["con"]  += L_con.item()
        totals["orth"] += L_o.item()

    nb = max(len(loader), 1)
    metrics = {k: v / nb for k, v in totals.items()}
    metrics["train_acc"] = n_correct / max(n_total, 1)
    return metrics


def build_optimizer(backbone, palmbridge, arcface):
    params = (list(backbone.parameters())
              + list(palmbridge.parameters())
              + list(arcface.parameters()))
    return optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer):
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return 0.1 + 0.9 * (epoch / max(WARMUP_EPOCHS - 1, 1))
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(backbone, palmbridge, arcface, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"epoch": epoch,
                "backbone":   backbone.state_dict(),
                "palmbridge": palmbridge.state_dict(),
                "arcface":    arcface.state_dict(),
                "optimizer":  optimizer.state_dict()}, path)
    print(f"  [ckpt] Saved → {path}")


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(backbone, palmbridge, loader, apply_palmbridge=True):
    backbone.eval(); palmbridge.eval()
    feats, lbls = [], []
    for imgs, labels in loader:
        z = backbone(imgs.to(DEVICE))
        if apply_palmbridge:
            z, _, _ = palmbridge(z)
        feats.append(z.cpu().numpy())
        lbls.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(lbls)


def compute_eer_metric(scores_array):
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return 1.0, 0.0
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return eer, thresh


def evaluate(backbone, palmbridge, probe_loader, gallery_loader,
             out_dir=".", tag="eval", apply_palmbridge=True):
    probe_feats,   probe_labels   = extract_features(
        backbone, palmbridge, probe_loader,   apply_palmbridge)
    gallery_feats, gallery_labels = extract_features(
        backbone, palmbridge, gallery_loader, apply_palmbridge)

    n_probe    = len(probe_feats)
    sim_matrix = probe_feats @ gallery_feats.T

    scores_list, labels_list = [], []
    for i in range(n_probe):
        for j in range(sim_matrix.shape[1]):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _     = compute_eer_metric(scores_arr)

    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]] for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")

    stats = palmbridge.codebook_usage()
    pb_str = "PB" if apply_palmbridge else "Naive"
    print(f"  [{tag}|{pb_str}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%  "
          f"cb_cos={stats['mean_cosine']:.3f}")
    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════

def run_experiment(train_samples, gallery_samples, probe_samples,
                   num_classes, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    train_loader   = make_loader(train_samples,   train=True,  shuffle=True)
    gallery_loader = make_loader(gallery_samples, train=False, shuffle=False)
    probe_loader   = make_loader(probe_samples,   train=False, shuffle=False)

    backbone   = CompNet(num_classes).to(DEVICE)
    palmbridge = PalmBridge().to(DEVICE)
    arcface    = ArcFaceLoss(num_classes).to(DEVICE)

    if PLUG_AND_PLAY and PB_CKPT:
        print(f"  [Plug-and-Play] loading PalmBridge from {PB_CKPT}")
        ckpt = torch.load(PB_CKPT, map_location=DEVICE)
        palmbridge.load_state_dict(ckpt["palmbridge"])
        print("  -- Naive baseline --")
        evaluate(backbone, palmbridge, probe_loader, gallery_loader,
                 rst_eval, "pnp_naive", apply_palmbridge=False)
        print("  -- PalmBridge plug-and-play --")
        eer, rank1 = evaluate(backbone, palmbridge, probe_loader, gallery_loader,
                              rst_eval, "pnp_pb", apply_palmbridge=True)
        return eer, rank1

    optimizer = build_optimizer(backbone, palmbridge, arcface)
    scheduler = build_scheduler(optimizer)
    best_eer  = float("inf")
    ckpt_path = os.path.join(results_dir, "net_params_best_eer.pt")

    # Pre-training baseline
    eer_pre, r1_pre = evaluate(backbone, palmbridge, probe_loader, gallery_loader,
                                rst_eval, "ep-001_pretrain", apply_palmbridge=False)
    best_eer = eer_pre

    for epoch in range(1, EPOCHS + 1):
        m = train_one_epoch(backbone, palmbridge, arcface, train_loader, optimizer, epoch)
        scheduler.step()

        phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "PalmBridge"
        print(f"  ep {epoch:03d}/{EPOCHS} [{phase}]  "
              f"loss={m['loss']:.4f}  bak={m['bak']:.4f}  "
              f"con={m['con']:.4f}  orth={m['orth']:.4f}  "
              f"acc={m['train_acc']*100:.2f}%")

        if epoch % LOG_EPOCHS == 0 or epoch == EPOCHS:
            use_pb = (epoch > WARMUP_EPOCHS)
            cur_eer, cur_rank1 = evaluate(
                backbone, palmbridge, probe_loader, gallery_loader,
                rst_eval, f"ep{epoch:04d}", apply_palmbridge=use_pb)
            if cur_eer < best_eer:
                best_eer = cur_eer
                save_checkpoint(backbone, palmbridge, arcface,
                                optimizer, epoch, ckpt_path)
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

    # Final evaluation: naive vs PalmBridge
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        backbone.load_state_dict(ckpt["backbone"])
        palmbridge.load_state_dict(ckpt["palmbridge"])

    print("  -- Naive baseline --")
    evaluate(backbone, palmbridge, probe_loader, gallery_loader,
             rst_eval, "FINAL_naive", apply_palmbridge=False)
    print("  -- PalmBridge --")
    final_eer, final_rank1 = evaluate(
        backbone, palmbridge, probe_loader, gallery_loader,
        rst_eval, "FINAL_pb", apply_palmbridge=True)

    return final_eer, final_rank1


# ══════════════════════════════════════════════════════════════
#  RESULTS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════

def print_and_save_summary(all_results, out_path):
    col_w  = 14
    header = (f"{'Setting':<22}"
              f"{'Train domain':<38}"
              f"{'Test domain':<26}"
              f"{'EER (%)':>{col_w}}"
              f"{'Rank-1 (%)':>{col_w}}")
    sep = "─" * len(header)
    lines = ["\nCross-Domain Open-Set Results — Palm-Auth (PalmBridge)",
             sep, header, sep]
    for r in all_results:
        eer_str   = f"{r['eer']:.2f}"   if r['eer']   is not None else "—"
        rank1_str = f"{r['rank1']:.2f}" if r['rank1'] is not None else "—"
        lines.append(f"{r['setting']:<22}"
                     f"{r['train_desc']:<38}"
                     f"{r['test_desc']:<26}"
                     f"{eer_str:>{col_w}}"
                     f"{rank1_str:>{col_w}}")
    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to: {out_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    device = DEVICE
    print(f"\n{'='*60}")
    print(f"  PalmBridge — Cross-Domain Open-Set (Palm-Auth)")
    print(f"  Protocol  : open set ({TRAIN_ID_RATIO*100:.0f}/{(1-TRAIN_ID_RATIO)*100:.0f} ID split, no overlap)")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {EPOCHS}  Warmup: {WARMUP_EPOCHS}")
    print(f"  K={NUM_PB_VECTORS}  W_MAP={W_MAP}  W_ORI={W_ORI}")
    print(f"  α={ALPHA}  β={BETA}  λ={LAMBDA_CON}")
    print(f"  Settings  : 2 scanner + {len(PAIRED_CONDITIONS)} paired-condition")
    print(f"  Results   : {SAVE_DIR}")
    print(f"{'='*60}")

    print("\n  Scanning dataset …")
    cond_paths    = _collect_perspective(PALM_AUTH_ROOT)
    scanner_paths = _collect_scanner(PALM_AUTH_ROOT, SCANNER_SPECTRA)
    print(f"  Perspective conditions found : {sorted(cond_paths.keys())}")
    print(f"  Scanner identities found     : {len(scanner_paths)}")

    # Load or generate shared disjoint train/test ID splits
    all_splits = load_or_generate_splits(
        cond_paths, scanner_paths, TRAIN_ID_RATIO, SEED)

    SETTINGS = []

    SETTINGS.append({
        "tag"        : "setting_scanner",
        "label"      : "S_scanner",
        "train_desc" : "Perspective (train IDs)",
        "test_desc"  : "Scanner (test IDs) 50/50",
        "parser"     : lambda: parse_setting_scanner(
                           cond_paths, scanner_paths,
                           all_splits["S_scanner"], SEED),
    })

    SETTINGS.append({
        "tag"        : "setting_scanner_to_persp",
        "label"      : "S_scanner_to_persp",
        "train_desc" : "Scanner (all scanner IDs)",
        "test_desc"  : "Perspective (no-scanner IDs)",
        "parser"     : lambda: parse_setting_scanner_to_perspective(
                           cond_paths, scanner_paths,
                           all_splits["S_scanner_to_persp"], SEED),
    })

    conditions_found = sorted(cond_paths.keys())
    for cond_a, cond_b in PAIRED_CONDITIONS:
        if cond_a not in conditions_found or cond_b not in conditions_found:
            print(f"  [WARN] '{cond_a}' or '{cond_b}' not found — skipping")
            continue
        ca, cb    = cond_a, cond_b
        split_key = f"S_{ca}_{cb}"
        if split_key not in all_splits:
            print(f"  [WARN] No split found for {split_key} — skipping")
            continue
        SETTINGS.append({
            "tag"        : f"setting_{ca}_{cb}",
            "label"      : f"S_{ca}_{cb}",
            "train_desc" : f"Perspective(not {ca}/{cb}) + Scanner",
            "test_desc"  : f"{ca}/{cb} (test IDs, random assign)",
            "parser"     : (lambda ca=ca, cb=cb: parse_setting_paired_conditions(
                                ca, cb, cond_paths, scanner_paths,
                                all_splits[f"S_{ca}_{cb}"], SEED)),
        })

    print(f"\n  Total settings to run : {len(SETTINGS)}")

    all_results = []

    for idx, s in enumerate(SETTINGS, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(SETTINGS)}] {s['label']}")
        print(f"  Train : {s['train_desc']}")
        print(f"  Test  : {s['test_desc']}")
        print(f"{'='*60}")

        results_dir = os.path.join(SAVE_DIR, s["tag"])
        t_start     = time.time()
        try:
            train_s, gal_s, probe_s, n_cls = s["parser"]()
            eer, rank1 = run_experiment(
                train_s, gal_s, probe_s, n_cls, results_dir)
            elapsed = time.time() - t_start
            print(f"\n  ✓  {s['label']}:  EER={eer*100:.4f}%  "
                  f"Rank-1={rank1:.2f}%  Time={elapsed/60:.1f} min")
            with open(os.path.join(results_dir, "results.json"), "w") as f:
                json.dump({"setting": s["label"], "train_desc": s["train_desc"],
                           "test_desc": s["test_desc"], "num_train_classes": n_cls,
                           "EER_pct": eer*100, "Rank1_pct": rank1}, f, indent=2)
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"],
                                 "eer": eer*100, "rank1": rank1})
        except Exception as e:
            print(f"\n  ✗  {s['label']} FAILED: {e}")
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"],
                                 "eer": None, "rank1": None})

    print(f"\n\n{'='*60}")
    print(f"  ALL {len(SETTINGS)} SETTINGS COMPLETE")
    print(f"{'='*60}")
    print_and_save_summary(
        all_results,
        os.path.join(SAVE_DIR, "results_summary.txt"))


if __name__ == "__main__":
    main()
