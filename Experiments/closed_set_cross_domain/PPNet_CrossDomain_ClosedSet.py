"""
PPNet — Cross-Domain Closed-Set Evaluations on X-Palm
========================================================
All evaluations follow a closed-set protocol: the same subject IDs
appear in both train and test splits. Gallery and probe are both drawn
from the TEST domain (not the training domain), split at the sample level.

Settings (12 total)
────────────────────
  S_scanner         │ Train : roi_perspective (all conditions, 190 IDs)
                    │ Gallery: 50% of scanner samples  (148 shared IDs)
                    │ Probe  : 50% of scanner samples  (148 shared IDs)

  S_scanner_to_persp│ Train : roi_scanner (148 IDs)
                    │ Gallery: 50% of perspective samples (148 shared IDs)
                    │ Probe  : 50% of perspective samples (148 shared IDs)

  S_(A,B) (×10)     │ Train : roi_perspective (all except A and B) + roi_scanner
                    │ Gallery: ALL condition A images  (first test domain)
                    │ Probe  : ALL condition B images  (second test domain)

Paired conditions:
  (wet,text) (wet,rnd) (rnd,text) (sf,roll) (jf,pitch)
  (bf,far) (roll,close) (far,jf) (fl,sf) (roll,pitch)

Scanner spectra kept: green | ir | yellow | pink | white

Model / loss: PPNet — CE + L2_reg + Contrastive + dis²
Matching metric: L2 distance on raw 512-d embeddings
EER: EER_all + EER_bal

Gallery/probe splits for S_scanner and S_scanner_to_persp are saved to
  x_palm_closedset_splits.json on first run and reused by all models.

Results saved to:
  {BASE_RESULTS_DIR}/setting_scanner/
  {BASE_RESULTS_DIR}/setting_{A}_{B}/
  {BASE_RESULTS_DIR}/results_summary.txt
"""


# ==============================================================
#  CONFIG
# ==============================================================
CONFIG = {
    "palm_auth_data_root"  : "path to X-Palm dataset",
    "scanner_spectra"      : {"green", "ir", "yellow", "pink", "white"},
    "test_gallery_ratio"   : 0.50,

    # Model (official PPNet)
    "img_side"             : 128,
    "dropout"              : 0.25,

    # Loss (official PPNet)
    "contrastive_margin"   : 5.0,
    "w_l2"                 : 1e-4,
    "w_contra"             : 2e-4,
    "w_dis"                : 1e-4,

    # Training
    "batch_size"           : 64,     # MUST be even
    "num_epochs"           : 200,
    "lr"                   : 0.0001,
    "lr_step"              : 17,
    "lr_gamma"             : 0.8,
    "augment_factor"       : 2,

    # Misc
    "base_results_dir"     : "./rst_ppnet_crossdomain",
    "random_seed"          : 42,
    "save_every"           : 50,
    "eval_every"           : 50,
    "num_workers"          : 4,
}

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

# Shared gallery/probe splits file — all models load this for fair comparison
SPLITS_FILE = "./x_palm_closedset_splits.json"

# ==============================================================

import os
import json
import math
import time
import random
import warnings
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ══════════════════════════════════════════════════════════════
#  MODEL  (exact official PPNet — unchanged)
# ══════════════════════════════════════════════════════════════

class ppnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv", nn.Conv2d(1, 16, 5, 1))
        self.layer1.add_module("bn",   nn.BatchNorm2d(16))

        self.layer2 = nn.Sequential()
        self.layer2.add_module("conv",    nn.Conv2d(16, 32, 1, 1))
        self.layer2.add_module("bn",      nn.BatchNorm2d(32, momentum=0.001))
        self.layer2.add_module("sigmoid", nn.Sigmoid())
        self.layer2.add_module("avgpool", nn.AvgPool2d(2, 2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module("conv",    nn.Conv2d(32, 64, 3, 1))
        self.layer3.add_module("bn",      nn.BatchNorm2d(64, momentum=0.001))
        self.layer3.add_module("sigmoid", nn.Sigmoid())
        self.layer3.add_module("avgpool", nn.AvgPool2d(2, 2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module("conv", nn.Conv2d(64, 64, 3, 1))
        self.layer4.add_module("bn",   nn.BatchNorm2d(64, momentum=0.001))
        self.layer4.add_module("relu", nn.ReLU())

        self.layer5 = nn.Sequential()
        self.layer5.add_module("conv",    nn.Conv2d(64, 256, 3, 1))
        self.layer5.add_module("bn",      nn.BatchNorm2d(256, momentum=0.001))
        self.layer5.add_module("relu",    nn.ReLU())
        self.layer5.add_module("maxpool", nn.MaxPool2d(2, 2))

        self.fc1   = nn.Linear(43264, 512)
        self.bn1   = nn.BatchNorm1d(512, momentum=0.001)
        self.relu1 = nn.ReLU()

        self.fc2   = nn.Linear(512, 512)
        self.bn2   = nn.BatchNorm1d(512, momentum=0.001)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.25)

        self.dis = nn.PairwiseDistance(p=2)
        self.fc3 = nn.Linear(512, num_classes)

    def _backbone(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.layer4(x); x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        return x

    def forward(self, x, y=None):
        x = self._backbone(x)
        b   = x.size(0)
        o1  = x[:b // 2, :]
        o2  = x[b // 2:, :]
        dis = self.dis(o1, o2)
        x   = self.drop2(x)
        x   = self.fc3(x)
        return x, dis

    @torch.no_grad()
    def get_embedding(self, x):
        """Raw (unnormalised) 512-d embedding for L2 distance matching."""
        return self._backbone(x)


# ══════════════════════════════════════════════════════════════
#  CONTRASTIVE LOSS  (exact official PPNet/train.py — unchanged)
# ══════════════════════════════════════════════════════════════

def contrastive_loss(target, dis, margin, device):
    n  = len(target) // 2
    y1 = target[:n]
    y2 = target[n:]
    y  = torch.zeros(n, device=device)
    y[y1 == y2] = 1.0
    margin_t = torch.full((n,), margin, device=device)
    return torch.mean(
        y * torch.pow(dis, 2)
        + (1 - y) * torch.pow(torch.clamp(margin_t - dis, min=0.0), 2))


# ══════════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    def __init__(self, outchannels=1): self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size(); tensor = tensor.view(c, h * w)
        idx = tensor > 0; t = tensor[idx]
        tensor[idx] = t.sub_(t.mean()).div_(t.std() + 1e-6)
        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, self.outchannels, dim=0)
        return tensor


# ══════════════════════════════════════════════════════════════
#  DATA COLLECTION HELPERS
# ══════════════════════════════════════════════════════════════

def _collect_perspective(data_root):
    """
    Returns cond_paths: condition → identity → [path, ...]
    Identity key: "{id}_{side_lowercase}"  e.g. "1_left"
    Filename: {id}_{side}_{condition}[_{n}].jpg → condition = parts[2].lower()
    "rnd" covers rnd_1…rnd_5 (parts[2] == "rnd" for all of them).
    """
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
    """
    Returns scanner_paths: identity → [path, ...]
    Filename: {id}_{Side}_{spectrum}_{rep}.jpg → lowercase side for key match.
    """
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
    """Flatten id2paths into a flat (path, label) list."""
    return [(p, label_map[ident])
            for ident, paths in id2paths.items()
            for p in paths]


# ══════════════════════════════════════════════════════════════
#  GALLERY/PROBE SPLIT PERSISTENCE
# ══════════════════════════════════════════════════════════════

def _gallery_probe_split_from_stored(id2paths, label_map, stored_split):
    """Reconstruct gallery/probe from stored path lists."""
    gallery, probe = [], []
    for ident, path_sets in stored_split.items():
        if ident not in label_map:
            continue
        label = label_map[ident]
        for p in path_sets["gallery"]:
            gallery.append((p, label))
        for p in path_sets["probe"]:
            probe.append((p, label))
    return gallery, probe


def _make_gallery_probe_split(id2paths, gallery_ratio, rng):
    """Create 50/50 sample-level split and return as storable dict."""
    stored = {}
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        n_gal = min(n_gal, len(paths) - 1) if len(paths) > 1 else len(paths)
        stored[ident] = {
            "gallery": paths[:n_gal],
            "probe"  : paths[n_gal:] if len(paths) > 1 else paths
        }
    return stored


def generate_closedset_splits(cond_paths, scanner_paths, gallery_ratio, seed):
    """
    Generate gallery/probe sample splits for scanner-based settings only.
    Paired-condition settings are deterministic (condA→gallery, condB→probe)
    and do not need to be stored.
    """
    rng = random.Random(seed)

    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    all_persp_ids = sorted(persp_all.keys())         # 190 IDs
    scanner_ids   = sorted(scanner_paths.keys())     # 148 IDs

    splits = {}

    # S_scanner: 50/50 split of scanner samples for the 148 shared IDs
    splits["S_scanner"] = _make_gallery_probe_split(
        {i: scanner_paths[i] for i in scanner_ids},
        gallery_ratio, rng)

    # S_scanner_to_persp: 50/50 split of perspective samples for the 148 shared IDs
    splits["S_scanner_to_persp"] = _make_gallery_probe_split(
        {i: persp_all[i] for i in scanner_ids},
        gallery_ratio, rng)

    return splits


def load_or_generate_closedset_splits(cond_paths, scanner_paths,
                                      gallery_ratio, seed):
    """
    Load splits from SPLITS_FILE if it exists; otherwise generate and save.
    Ensures all models use identical gallery/probe sample assignments.
    """
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE) as f:
            splits = json.load(f)
        print(f"  Loaded existing gallery/probe splits from: {SPLITS_FILE}")
    else:
        print(f"  Generating gallery/probe splits (seed={seed}) → {SPLITS_FILE}")
        splits = generate_closedset_splits(
            cond_paths, scanner_paths, gallery_ratio, seed)
        with open(SPLITS_FILE, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"  Splits saved to: {SPLITS_FILE}")

    for key, val in splits.items():
        n_ids = len(val)
        n_gal = sum(len(v["gallery"]) for v in val.values())
        n_prb = sum(len(v["probe"])   for v in val.values())
        print(f"    {key:<30}  IDs={n_ids}  gallery={n_gal}  probe={n_prb}")

    return splits


# ══════════════════════════════════════════════════════════════
#  PARSERS FOR EACH SETTING
# ══════════════════════════════════════════════════════════════

def _gallery_probe_split(id2paths, label_map, gallery_ratio, rng):
    """50/50 sample-level split — every ID appears in both gallery and probe."""
    gallery, probe = [], []
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        n_gal = min(n_gal, len(paths) - 1) if len(paths) > 1 else len(paths)
        for p in paths[:n_gal]: gallery.append((p, label_map[ident]))
        for p in paths[n_gal:]: probe.append((p, label_map[ident]))
    return gallery, probe


def parse_setting_scanner(cond_paths, scanner_paths, stored_splits, seed):
    """
    S_scanner — Perspective (train) → Scanner (gallery + probe)
    ─────────────────────────────────────────────────────────────
    Train   : ALL perspective images for ALL 190 perspective IDs
    Gallery : 50% of scanner samples for the 148 shared IDs (pre-computed)
    Probe   : 50% of scanner samples for the 148 shared IDs (pre-computed)
    Closed-set: 148 test IDs are a subset of the 190 training IDs.
    """
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    all_persp_ids = sorted(persp_all.keys())   # 190
    scanner_ids   = sorted(scanner_paths.keys())  # 148

    # Training label map uses ALL 190 perspective IDs
    train_label_map = {ident: i for i, ident in enumerate(all_persp_ids)}
    # Test label map uses only the 148 scanner IDs
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    num_train_cls   = len(all_persp_ids)

    train_samples = [(p, train_label_map[i])
                     for i in all_persp_ids for p in persp_all[i]]

    split = stored_splits["S_scanner"]
    gallery_samples, probe_samples = _gallery_probe_split_from_stored(
        {i: scanner_paths[i] for i in scanner_ids}, test_label_map, split)

    _print_stats("S_scanner | Perspective/190 (train) → Scanner/148 50/50 (test)",
                 len(all_persp_ids), len(scanner_ids), len(train_samples),
                 len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_scanner_to_perspective(cond_paths, scanner_paths,
                                         stored_splits, seed):
    """
    S_scanner_to_persp — Scanner (train) → Perspective (gallery + probe)
    ──────────────────────────────────────────────────────────────────────
    Train   : ALL scanner images for ALL 148 scanner IDs
    Gallery : 50% of perspective samples for the 148 shared IDs (pre-computed)
    Probe   : 50% of perspective samples for the 148 shared IDs (pre-computed)
    Closed-set: same 148 IDs in train and test.
    """
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    scanner_ids = sorted(scanner_paths.keys())  # 148

    train_label_map = {ident: i for i, ident in enumerate(scanner_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    num_train_cls   = len(scanner_ids)

    train_samples = [(p, train_label_map[i])
                     for i in scanner_ids for p in scanner_paths[i]]

    split = stored_splits["S_scanner_to_persp"]
    gallery_samples, probe_samples = _gallery_probe_split_from_stored(
        {i: persp_all[i] for i in scanner_ids}, test_label_map, split)

    _print_stats("S_scanner_to_persp | Scanner/148 (train) → Perspective/148 50/50 (test)",
                 len(scanner_ids), len(scanner_ids), len(train_samples),
                 len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_paired_conditions(cond_a, cond_b, cond_paths, scanner_paths, seed):
    """
    S_{A}_{B} — Paired-condition closed-set
    ─────────────────────────────────────────
    All IDs with BOTH condition A and B are used (closed-set, no ID split).
    Train   : scanner + perspective (all except cond_A and cond_B)
    Gallery : ALL condition A images   (first test domain)
    Probe   : ALL condition B images   (second test domain)
    """
    paths_a = cond_paths.get(cond_a, {})
    paths_b = cond_paths.get(cond_b, {})
    if not paths_a:
        raise ValueError(f"No images for condition '{cond_a}'")
    if not paths_b:
        raise ValueError(f"No images for condition '{cond_b}'")

    eligible_ids = sorted(set(paths_a.keys()) & set(paths_b.keys()))
    if not eligible_ids:
        raise ValueError(f"No IDs with both '{cond_a}' and '{cond_b}'")

    label_map   = {ident: i for i, ident in enumerate(eligible_ids)}
    num_classes = len(eligible_ids)

    train_samples = []
    for cond, cond_dict in cond_paths.items():
        if cond in (cond_a, cond_b):
            continue
        for ident in eligible_ids:
            for p in cond_dict.get(ident, []):
                train_samples.append((p, label_map[ident]))
    for ident in eligible_ids:
        for p in scanner_paths.get(ident, []):
            train_samples.append((p, label_map[ident]))

    gallery_samples = _all_samples(
        {ident: paths_a[ident] for ident in eligible_ids}, label_map)
    probe_samples   = _all_samples(
        {ident: paths_b[ident] for ident in eligible_ids}, label_map)

    _print_stats(
        f"S_{cond_a}_{cond_b} | Perspective(not {cond_a}/{cond_b})+Scanner"
        f" → gallery:{cond_a} / probe:{cond_b}",
        num_classes, num_classes, len(train_samples),
        len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_classes


def _print_stats(name, n_train_ids, n_test_ids, train_n, gallery_n, probe_n):
    print(f"\n  [{name}]")
    print(f"    Train IDs / Test IDs  : {n_train_ids} / {n_test_ids}")
    print(f"    Train images          : {train_n}")
    print(f"    Gallery / Probe       : {gallery_n} / {probe_n}")


# ══════════════════════════════════════════════════════════════
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(net, num_classes, cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    model_name   = type(net.module if isinstance(net, DataParallel) else net).__name__
    weights_path = os.path.join(cache_dir,
                                f"init_weights_{model_name}_nc{num_classes}.pth")
    _net = net.module if isinstance(net, DataParallel) else net
    if os.path.exists(weights_path):
        print(f"  Loading cached init weights: {weights_path}")
        _net.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"  Saving init weights: {weights_path}")
        torch.save(_net.state_dict(), weights_path)
    return net


# ══════════════════════════════════════════════════════════════
#  PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════

class AugmentedDataset(Dataset):
    def __init__(self, samples, img_side=128, augment_factor=1):
        self.samples        = samples
        self.augment_factor = augment_factor
        self.aug_transform  = T.Compose([
            T.Resize(img_side),
            T.RandomChoice([
                T.ColorJitter(brightness=0.3, contrast=0.3),
                T.RandomResizedCrop(img_side, scale=(0.9,1.0), ratio=(1.0,1.0)),
                T.RandomRotation(degrees=8, expand=False),
                T.RandomPerspective(distortion_scale=0.15, p=0.8),
            ]),
            T.ToTensor(), NormSingleROI(outchannels=1),
        ])

    def __len__(self): return len(self.samples) * self.augment_factor

    def __getitem__(self, index):
        real_idx    = index % len(self.samples)
        path, label = self.samples[real_idx]
        return self.aug_transform(Image.open(path).convert("L")), label


class SingleDataset(Dataset):
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([T.Resize(img_side), T.ToTensor(),
                                    NormSingleROI(outchannels=1)])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


# ══════════════════════════════════════════════════════════════
#  TRAINING  (PPNet composite loss — unchanged from official)
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, optimizer, device, phase,
                  margin=5.0, w_l2=1e-4, w_contra=2e-4, w_dis=1e-4):
    """
    PPNet composite loss:
      CE + w_l2*L2_reg(fc2,fc3) + w_contra*ContrastiveLoss + w_dis*mean(dis²)
    Batch must be even; an odd last batch is padded by duplicating first sample.
    """
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for data, target in loader:
            if len(target) % 2 != 0:
                target = torch.cat((target, target[0:1]), dim=0)
                data   = torch.cat((data,   data[0:1]),   dim=0)

            data, target = data.to(device), target.to(device)
            if is_train: optimizer.zero_grad()

            output, dis = model(data)

            cross  = criterion(output, target)
            _m     = model.module if isinstance(model, DataParallel) else model
            l2_reg = torch.norm(_m.fc2.weight, 2) + torch.norm(_m.fc3.weight, 2)
            contra = contrastive_loss(target, dis, margin, device)
            loss   = cross + w_l2*l2_reg + w_contra*contra + w_dis*torch.mean(dis**2)

            if is_train: loss.backward(); optimizer.step()

            running_loss    += loss.item() * data.size(0)
            running_correct += output.data.max(1)[1].eq(target).sum().item()
            total           += data.size(0)

    return running_loss / max(total, 1), 100.0 * running_correct / max(total, 1)


# ══════════════════════════════════════════════════════════════
#  EVALUATION  (L2 distance, EER_all + EER_bal, argmin Rank-1)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval(); feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _single_eer(genuine, impostor):
    """EER from genuine/impostor distance arrays (lower = more similar → flip)."""
    if genuine.mean() < impostor.mean():
        genuine = -genuine; impostor = -impostor
    y   = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    s   = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)


def compute_eer(scores_array, n_trials=10, seed=42):
    """
    scores_array[:,0] = L2 distance (lower = more similar)
    scores_array[:,1] = +1 genuine | -1 impostor
    Returns (eer_all, eer_bal).
    """
    rng  = np.random.RandomState(seed)
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return 1.0, 1.0
    eer_all = _single_eer(ins.copy(), outs.copy())
    n_imp   = min(len(ins), len(outs))
    eers    = [_single_eer(ins.copy(), rng.choice(outs, size=n_imp, replace=False))
               for _ in range(n_trials)]
    return eer_all, float(np.mean(eers))


def evaluate(model, probe_loader, gallery_loader, device,
             out_dir=".", tag="eval"):
    """L2-distance evaluation. Returns (eer_all, eer_bal, rank1)."""
    probe_feats,   probe_labels   = extract_features(model, probe_loader,   device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe   = len(probe_feats)
    n_gallery = len(gallery_feats)

    probe_sq   = np.sum(probe_feats   ** 2, axis=1, keepdims=True)
    gallery_sq = np.sum(gallery_feats ** 2, axis=1, keepdims=True).T
    dot        = probe_feats @ gallery_feats.T
    dist_matrix = np.sqrt(np.maximum(probe_sq + gallery_sq - 2 * dot, 0.0))

    scores_list, labels_list = [], []
    for i in range(n_probe):
        for j in range(n_gallery):
            scores_list.append(float(dist_matrix[i, j]))
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr       = np.column_stack([scores_list, labels_list])
    eer_all, eer_bal = compute_eer(scores_arr)

    nn_idx  = np.argmin(dist_matrix, axis=1)
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]] for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")

    print(f"  [{tag}]  EER_all={eer_all*100:.4f}%  "
          f"EER_bal={eer_bal*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer_all, eer_bal, rank1


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════

def run_experiment(train_samples, gallery_samples, probe_samples,
                   num_classes, cfg, results_dir, device):
    """Train PPNet and evaluate. Returns (final_eer_bal, final_rank1)."""
    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    img_side       = cfg["img_side"]
    batch_size     = cfg["batch_size"]
    num_epochs     = cfg["num_epochs"]
    augment_factor = cfg["augment_factor"]
    nw             = cfg["num_workers"]
    eval_every     = cfg["eval_every"]
    save_every     = cfg["save_every"]
    margin         = cfg["contrastive_margin"]
    w_l2           = cfg["w_l2"]
    w_contra       = cfg["w_contra"]
    w_dis          = cfg["w_dis"]

    assert batch_size % 2 == 0, f"batch_size must be even, got {batch_size}"

    train_loader = DataLoader(
        AugmentedDataset(train_samples, img_side, augment_factor),
        batch_size=batch_size, shuffle=True, num_workers=nw,
        pin_memory=True, drop_last=True)
    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    net = ppnet(num_classes=num_classes)
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    net = get_or_create_init_weights(
        net, num_classes,
        cache_dir = cfg["base_results_dir"],
        device    = device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg["lr"])
    scheduler = lr_scheduler.StepLR(optimizer, cfg["lr_step"], cfg["lr_gamma"])

    # ── Pre-training baseline ─────────────────────────────────────────────
    _net = net.module if isinstance(net, DataParallel) else net
    pre_eer_all, pre_eer_bal, pre_r1 = evaluate(
        _net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag="ep-001_pretrain")
    best_eer     = pre_eer_bal
    last_eer_all = pre_eer_all; last_eer_bal = pre_eer_bal; last_rank1 = pre_r1
    torch.save(_net.state_dict(),
               os.path.join(results_dir, "net_params_best_eer.pth"))

    train_losses, train_accs = [], []

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            net, train_loader, criterion, optimizer, device, "training",
            margin=margin, w_l2=w_l2, w_contra=w_contra, w_dis=w_dis)
        scheduler.step()
        train_losses.append(t_loss); train_accs.append(t_acc)
        _net = net.module if isinstance(net, DataParallel) else net

        if (epoch % eval_every == 0 and epoch > 0) or epoch == num_epochs - 1:
            cur_eer_all, cur_eer_bal, cur_rank1 = evaluate(
                _net, probe_loader, gallery_loader,
                device, out_dir=rst_eval, tag=f"ep{epoch:04d}")
            last_eer_all = cur_eer_all; last_eer_bal = cur_eer_bal
            last_rank1   = cur_rank1
            if cur_eer_bal < best_eer:
                best_eer = cur_eer_bal
                torch.save(_net.state_dict(),
                           os.path.join(results_dir, "net_params_best_eer.pth"))
                print(f"  *** New best EER_bal: {best_eer*100:.4f}% ***")

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ts = time.strftime("%H:%M:%S")
            eer_str = (f"EER_all={last_eer_all*100:.4f}% | "
                       f"EER_bal={last_eer_bal*100:.4f}%"
                       if not math.isnan(last_eer_all) else "N/A")
            rank1_str = f"{last_rank1:.2f}%" if not math.isnan(last_rank1) else "N/A"
            print(f"  [{ts}] ep {epoch:04d} | loss={t_loss:.4f} | acc={t_acc:.2f}% | "
                  f"{eer_str} | Rank-1={rank1_str}")

        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))

    # ── Final evaluation (best checkpoint) ────────────────────────────────
    best_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(results_dir, "net_params.pth")
    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_path, map_location=device))
    final_eer_all, final_eer_bal, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag="FINAL")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(train_losses, 'b'); axes[0].set_title("Train Loss")
        axes[0].set_xlabel("epoch"); axes[0].grid(True)
        axes[1].plot(train_accs,   'b'); axes[1].set_title("Train Acc (%)")
        axes[1].set_xlabel("epoch"); axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "train_curves.png"))
        plt.close(fig)
    except Exception:
        pass

    return final_eer_bal, final_rank1


# ══════════════════════════════════════════════════════════════
#  RESULTS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════

def print_and_save_summary(all_results, out_path):
    col_w  = 14
    header = (f"{'Setting':<22}"
              f"{'Train domain':<38}"
              f"{'Test domain':<26}"
              f"{'EER_bal (%)':>{col_w}}"
              f"{'Rank-1 (%)':>{col_w}}")
    sep = "─" * len(header)
    lines = ["\nCross-Domain Closed-Set Results — Palm-Auth (PPNet)", sep, header, sep]

    for r in all_results:
        eer_str   = f"{r['eer_bal']:.2f}"  if r['eer_bal'] is not None else "—"
        rank1_str = f"{r['rank1']:.2f}"    if r['rank1']   is not None else "—"
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
    cfg  = CONFIG
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    assert cfg["batch_size"] % 2 == 0, \
        f"batch_size must be even for PPNet contrastive pairing, got {cfg['batch_size']}"

    device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir = cfg["base_results_dir"]
    os.makedirs(base_results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PPNet — Cross-Domain Closed-Set (Palm-Auth)")
    print(f"  Protocol : closed set (shared IDs in train & test)")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {cfg['num_epochs']}")
    print(f"  Loss     : CE + {cfg['w_l2']}*L2 + "
          f"{cfg['w_contra']}*Contra(m={cfg['contrastive_margin']}) + "
          f"{cfg['w_dis']}*dis²")
    print(f"  Matching : L2 distance (unnormalised 512-d embeddings)")
    print(f"  Settings  : 2 scanner + 10 paired-condition")
    print(f"  Results  : {base_results_dir}")
    print(f"{'='*60}")

    # ── Pre-collect data once ─────────────────────────────────────────────
    print("\n  Scanning dataset …")
    cond_paths    = _collect_perspective(cfg["palm_auth_data_root"])
    scanner_paths = _collect_scanner(cfg["palm_auth_data_root"],
                                     cfg["scanner_spectra"])
    print(f"  Perspective conditions found : {sorted(cond_paths.keys())}")
    print(f"  Scanner identities found     : {len(scanner_paths)}")

    # ── Build settings list ───────────────────────────────────────────────
    # ── Load or generate shared gallery/probe splits ─────────────────────────
    all_splits = load_or_generate_closedset_splits(
        cond_paths, scanner_paths, cfg["test_gallery_ratio"], seed)

    # ── Build settings list ───────────────────────────────────────────────────
    SETTINGS = []

    SETTINGS.append({
        "tag"        : "setting_scanner",
        "label"      : "S_scanner",
        "train_desc" : "Perspective (all 190 IDs)",
        "test_desc"  : "Scanner 50/50 gallery/probe (148 IDs)",
        "parser"     : lambda: parse_setting_scanner(
                           cond_paths, scanner_paths, all_splits, seed),
    })

    SETTINGS.append({
        "tag"        : "setting_scanner_to_persp",
        "label"      : "S_scanner_to_persp",
        "train_desc" : "Scanner (all 148 IDs)",
        "test_desc"  : "Perspective 50/50 gallery/probe (148 IDs)",
        "parser"     : lambda: parse_setting_scanner_to_perspective(
                           cond_paths, scanner_paths, all_splits, seed),
    })

    conditions_found = sorted(cond_paths.keys())
    for cond_a, cond_b in PAIRED_CONDITIONS:
        if cond_a not in conditions_found or cond_b not in conditions_found:
            print(f"  [WARN] '{cond_a}' or '{cond_b}' not found — skipping")
            continue
        ca, cb = cond_a, cond_b
        SETTINGS.append({
            "tag"        : f"setting_{ca}_{cb}",
            "label"      : f"S_{ca}_{cb}",
            "train_desc" : f"Perspective(not {ca}/{cb}) + Scanner",
            "test_desc"  : f"gallery:{ca} / probe:{cb}",
            "parser"     : (lambda ca=ca, cb=cb: parse_setting_paired_conditions(
                                ca, cb, cond_paths, scanner_paths, seed)),
        })

        print(f"\n  Total settings to run : {len(SETTINGS)}")

    # ── Run all settings ──────────────────────────────────────────────────
    all_results = []

    for idx, s in enumerate(SETTINGS, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(SETTINGS)}] {s['label']}")
        print(f"  Train : {s['train_desc']}")
        print(f"  Test  : {s['test_desc']}")
        print(f"{'='*60}")

        results_dir = os.path.join(base_results_dir, s["tag"])
        t_start     = time.time()
        try:
            train_s, gal_s, probe_s, n_cls = s["parser"]()
            eer_bal, rank1 = run_experiment(
                train_s, gal_s, probe_s, n_cls, cfg, results_dir, device)
            elapsed = time.time() - t_start
            print(f"\n  ✓  {s['label']}:  EER_bal={eer_bal*100:.4f}%  "
                  f"Rank-1={rank1:.2f}%  Time={elapsed/60:.1f} min")
            with open(os.path.join(results_dir, "results.json"), "w") as f:
                json.dump({"setting"     : s["label"],
                           "train_desc"  : s["train_desc"],
                           "test_desc"   : s["test_desc"],
                           "num_classes" : n_cls,
                           "EER_bal_pct" : eer_bal * 100,
                           "Rank1_pct"   : rank1}, f, indent=2)
            all_results.append({"setting"    : s["label"],
                                 "train_desc" : s["train_desc"],
                                 "test_desc"  : s["test_desc"],
                                 "eer_bal"    : eer_bal * 100,
                                 "rank1"      : rank1})
        except Exception as e:
            print(f"\n  ✗  {s['label']} FAILED: {e}")
            all_results.append({"setting"    : s["label"],
                                 "train_desc" : s["train_desc"],
                                 "test_desc"  : s["test_desc"],
                                 "eer_bal"    : None,
                                 "rank1"      : None})

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"  ALL {len(SETTINGS)} SETTINGS COMPLETE")
    print(f"{'='*60}")
    print_and_save_summary(
        all_results,
        os.path.join(base_results_dir, "results_summary.txt"))


if __name__ == "__main__":
    main()
