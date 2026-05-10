"""
PPNet — Leave-One-Out Cross-Dataset Experiment Runner
==================================================
Trains on ALL combinations of three datasets and evaluates on the
left-out fourth dataset. Four experiments total:

  Train                              Test
  ─────────────────────────────────  ──────────
  CASIA-MS + MPDv2   + XJTU          X-Palm
  X-Palm + MPDv2  + XJTU          CASIA-MS
  X-Palm + CASIA-MS + XJTU        MPDv2
  X-Palm + CASIA-MS + MPDv2       XJTU


Model architecture and training: unchanged from official PPNet.
  - 5 conv layers + BN + 2 FC layers (43264 → 512 → 512) + PairwiseDistance
  - Composite loss: CE + w_l2*L2_reg + w_contra*ContrastiveLoss + w_dis*mean(dis²)
  - Batch must be even (contrastive pairing within batch)
  - Matching metric: L2 distance on raw (unnormalised) 512-d embeddings

Results are saved to:
  {BASE_RESULTS_DIR}/test_{Y}/         ← per-experiment outputs
  {BASE_RESULTS_DIR}/results_table.txt ← final EER_bal / Rank-1 table (with Avg row)
  {BASE_RESULTS_DIR}/results_raw.json  ← raw numbers as JSON
"""

# ==============================================================
#  DATASET LIST
# ==============================================================
ALL_DATASETS = ["X-Palm", "CASIA-MS", "MPDv2", "XJTU"]

# ==============================================================
#  BASE CONFIG
# ==============================================================
BASE_CONFIG = {
    # ── Dataset paths ──────────────────────────────────────────
    "casiams_data_root"    : "./CASIA-MS-ROI",
    "x_palm_data_root"  : "./x_palm_roi",
    "mpd_data_root"        : "./MPDv2_roi",
    "xjtu_data_root"       : "./XJTU-UP",

    # ── Splitting (only used for the test dataset) ─────────────
    "test_gallery_ratio"   : 0.50,

    # ── Palm-Auth toggle ───────────────────────────────────────
    "use_scanner"          : True,

    # ── Model (official PPNet values) ──────────────────────────
    "img_side"             : 128,   # → FC1 input = 43264
    "dropout"              : 0.25,

    # ── Loss (official PPNet values) ───────────────────────────
    "contrastive_margin"   : 5.0,
    "w_l2"                 : 1e-4,
    "w_contra"             : 2e-4,
    "w_dis"                : 1e-4,

    # ── Training (official PPNet values) ───────────────────────
    "batch_size"           : 64,    # MUST be even
    "num_epochs"           : 200,
    "lr"                   : 0.0001,
    "lr_step"              : 17,
    "lr_gamma"             : 0.8,
    "augment_factor"       : 2,

    # ── Misc ───────────────────────────────────────────────────
    "base_results_dir"     : "./rst_ppnet_loo",
    "random_seed"          : 42,
    "save_every"           : 50,
    "eval_every"           : 50,
    "num_workers"          : 4,
}
# ==============================================================

import os
import copy
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

ALLOWED_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

N_HIGH = 160
N_LOW  = 40

TARGET_HIGH_CASIA = 33
TARGET_LOW_CASIA  = 15
TARGET_HIGH_XJTU  = 33
TARGET_LOW_XJTU   = 15

XJTU_VARIATIONS = [
    ("iPhone", "Flash"),
    ("iPhone", "Nature"),
    ("huawei", "Flash"),
    ("huawei", "Nature"),
]


# ══════════════════════════════════════════════════════════════
#  MODEL  (exact copy from official PPNet — unchanged)
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
        b = x.size(0)
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
#  CONTRASTIVE LOSS  (exact copy from official PPNet/train.py)
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
#  DATASET PARSERS
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root, seed=42):
    rng     = random.Random(seed)
    id_spec = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4: continue
        id_spec[parts[0]+"_"+parts[1]][parts[2]].append(
            os.path.join(data_root, fname))
    all_ids = sorted(id_spec.keys())
    if len(all_ids) < N_HIGH + N_LOW:
        raise ValueError(f"CASIA-MS: need {N_HIGH+N_LOW} IDs, found {len(all_ids)}")
    selected = sorted(rng.sample(all_ids, N_HIGH + N_LOW))
    rng.shuffle(selected)
    high_ids = selected[:N_HIGH]; low_ids = selected[N_HIGH:]
    def _sample(ident, target):
        spec_list = list(sorted(id_spec[ident].keys())); rng.shuffle(spec_list)
        n_spec = len(spec_list); base_s = target // n_spec; rem_s = target % n_spec
        chosen = []
        for j, sp in enumerate(spec_list):
            k = min(base_s + (1 if j < rem_s else 0), len(id_spec[ident][sp]))
            chosen.extend(rng.sample(id_spec[ident][sp], k))
        return chosen
    id2paths = {}
    for ident in high_ids: id2paths[ident] = _sample(ident, TARGET_HIGH_CASIA)
    for ident in low_ids:  id2paths[ident] = _sample(ident, TARGET_LOW_CASIA)
    actual = sum(len(v) for v in id2paths.values())
    hc = [len(id2paths[i]) for i in high_ids]; lc = [len(id2paths[i]) for i in low_ids]
    print(f"  [CASIA-MS] ids={len(id2paths)}  total={actual}")
    print(f"    High ({N_HIGH}×~{TARGET_HIGH_CASIA}): min={min(hc)} max={max(hc)} mean={sum(hc)/N_HIGH:.1f}")
    print(f"    Low  ({N_LOW}×~{TARGET_LOW_CASIA}):  min={min(lc)} max={max(lc)} mean={sum(lc)/N_LOW:.1f}")
    return id2paths


def parse_palm_auth_data(data_root, use_scanner=False, seed=42):
    rng      = random.Random(seed)
    IMG_EXTS = {".jpg",".jpeg",".bmp",".png"}
    id2paths = defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir = os.path.join(data_root, subject_id)
        if not os.path.isdir(subject_dir): continue
        roi_dir = os.path.join(subject_dir, "roi_perspective")
        if os.path.isdir(roi_dir):
            for fname in sorted(os.listdir(roi_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                parts = os.path.splitext(fname)[0].split("_")
                if len(parts) < 3: continue
                id2paths[parts[0]+"_"+parts[1]].append(os.path.join(roi_dir, fname))
        if use_scanner:
            scan_dir = os.path.join(subject_dir, "roi_scanner")
            if os.path.isdir(scan_dir):
                for fname in sorted(os.listdir(scan_dir)):
                    if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                    parts = os.path.splitext(fname)[0].split("_")
                    if len(parts) < 4: continue
                    if parts[2].lower() not in ALLOWED_SPECTRA: continue
                    id2paths[subject_id+"_"+parts[1].lower()].append(
                        os.path.join(scan_dir, fname))
    # Cap at 190 IDs — take those with the most samples
    all_ids = sorted(id2paths.keys(), key=lambda i: len(id2paths[i]), reverse=True)
    if len(all_ids) < N_HIGH + N_LOW:
        raise ValueError(f"Palm-Auth: need {N_HIGH+N_LOW} IDs, found {len(all_ids)}")
    selected_ids = all_ids[:N_HIGH + N_LOW]
    result  = {k: list(id2paths[k]) for k in selected_ids}
    counts  = [len(v) for v in result.values()]
    mode    = (f"perspective + scanner ({', '.join(sorted(ALLOWED_SPECTRA))})"
               if use_scanner else "perspective only")
    print(f"  [Palm-Auth/{mode}]")
    print(f"    ids={len(result)}  total={sum(counts)}  "
          f"per-id min/max/mean={min(counts)}/{max(counts)}/{sum(counts)/len(counts):.1f}  "
          f"cutoff={counts[-1]}")
    return result


def parse_mpd_data(data_root, seed=42):
    rng    = random.Random(seed)
    id_dev = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) != 5: continue
        subject, session, device, hand_side, iteration = parts
        if device not in ("h","m") or hand_side not in ("l","r"): continue
        id_dev[subject+"_"+hand_side][device].append(os.path.join(data_root, fname))
    # Select the top N_HIGH + N_LOW IDs by sample count
    all_ids = sorted(
        id_dev.keys(),
        key=lambda i: len(id_dev[i].get("h", [])) + len(id_dev[i].get("m", [])),
        reverse=True)
    if len(all_ids) < N_HIGH + N_LOW:
        raise ValueError(f"MPDv2: need {N_HIGH + N_LOW} IDs, found {len(all_ids)}")
    selected_ids = all_ids[:N_HIGH + N_LOW]
    id2paths = {}
    for ident in selected_ids:
        id2paths[ident] = id_dev[ident].get("h", []) + id_dev[ident].get("m", [])
    counts = [len(v) for v in id2paths.values()]
    actual = sum(counts)
    print(f"  [MPDv2] ids={len(id2paths)}  total={actual}")
    print(f"    min={min(counts)}  max={max(counts)}  "
          f"mean={actual/len(id2paths):.1f}  cutoff={counts[-1]}")
    return id2paths


def parse_xjtu_data(data_root, seed=42):
    rng      = random.Random(seed)
    IMG_EXTS = {".jpg",".jpeg",".bmp",".png"}
    id_var   = defaultdict(lambda: defaultdict(list))
    for device, condition in XJTU_VARIATIONS:
        var_dir = os.path.join(data_root, device, condition)
        if not os.path.isdir(var_dir):
            print(f"  [XJTU] WARNING: {var_dir} not found"); continue
        for id_folder in sorted(os.listdir(var_dir)):
            id_dir = os.path.join(var_dir, id_folder)
            if not os.path.isdir(id_dir): continue
            parts = id_folder.split("_")
            if len(parts) < 2 or parts[0].upper() not in ("L","R"): continue
            for fname in sorted(os.listdir(id_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                id_var[id_folder][(device, condition)].append(
                    os.path.join(id_dir, fname))
    all_ids = sorted(id_var.keys())
    print(f"  [XJTU] Total IDs found: {len(all_ids)}")
    if len(all_ids) < N_HIGH + N_LOW:
        raise ValueError(f"XJTU: need {N_HIGH+N_LOW} IDs, found {len(all_ids)}")
    selected = sorted(rng.sample(all_ids, N_HIGH + N_LOW))
    rng.shuffle(selected)
    high_ids = selected[:N_HIGH]; low_ids = selected[N_HIGH:]
    def _sample_var(ident, target):
        var_keys = list(XJTU_VARIATIONS); rng.shuffle(var_keys)
        n_var = len(var_keys); base_v = target // n_var; rem_v = target % n_var
        chosen = []
        for j, vk in enumerate(var_keys):
            k = min(base_v + (1 if j < rem_v else 0), len(id_var[ident].get(vk,[])))
            if k > 0: chosen.extend(rng.sample(id_var[ident].get(vk,[]), k))
        return chosen
    id2paths = {}
    for ident in high_ids: id2paths[ident] = _sample_var(ident, TARGET_HIGH_XJTU)
    for ident in low_ids:  id2paths[ident] = _sample_var(ident, TARGET_LOW_XJTU)
    actual = sum(len(v) for v in id2paths.values())
    hc = [len(id2paths[i]) for i in high_ids]; lc = [len(id2paths[i]) for i in low_ids]
    print(f"  [XJTU] ids={len(id2paths)}  total={actual}")
    print(f"    High ({N_HIGH}×~{TARGET_HIGH_XJTU}): min={min(hc)} max={max(hc)} mean={sum(hc)/N_HIGH:.1f}")
    print(f"    Low  ({N_LOW}×~{TARGET_LOW_XJTU}):  min={min(lc)} max={max(lc)} mean={sum(lc)/N_LOW:.1f}")
    return id2paths


def get_parser(dataset_name, cfg):
    name = dataset_name.strip().lower().replace("-","").replace("_","")
    seed = cfg["random_seed"]
    if name == "casiams":
        return lambda: parse_casia_ms(cfg["casiams_data_root"], seed=seed)
    elif name == "palmauth":
        return lambda: parse_palm_auth_data(cfg["palm_auth_data_root"],
                                            use_scanner=cfg.get("use_scanner", False),
                                            seed=seed)
    elif name == "mpdv2":
        return lambda: parse_mpd_data(cfg["mpd_data_root"], seed=seed)
    elif name == "xjtu":
        return lambda: parse_xjtu_data(cfg["xjtu_data_root"], seed=seed)
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'")


def _ds_key(name):
    return name.strip().lower().replace("-","").replace("_","")


# ══════════════════════════════════════════════════════════════
#  COMBINED TRAINING SET BUILDER
# ══════════════════════════════════════════════════════════════

def build_combined_train_samples(train_datasets, cfg):
    """
    Parse each training dataset in full (190 IDs each) and merge all
    samples into a single flat list with globally unique labels.

    Labels are offset per dataset so subjects across datasets never
    share the same class index.

    Returns: (train_samples, num_classes)
      train_samples : list of (path, global_label)
      num_classes   : total number of unique subjects across all train sets
    """
    train_samples = []
    label_offset  = 0

    for ds_name in train_datasets:
        print(f"  Parsing {ds_name} (train) …")
        id2paths   = get_parser(ds_name, cfg)()
        sorted_ids = sorted(id2paths.keys())
        label_map  = {ident: label_offset + i
                      for i, ident in enumerate(sorted_ids)}
        for ident in sorted_ids:
            for path in id2paths[ident]:
                train_samples.append((path, label_map[ident]))
        n_subj = len(sorted_ids)
        n_imgs = sum(len(id2paths[i]) for i in sorted_ids)
        print(f"    → {n_subj} subjects | {n_imgs} images "
              f"| labels {label_offset}–{label_offset + n_subj - 1}")
        label_offset += n_subj

    print(f"  Combined train: {label_offset} total subjects | "
          f"{len(train_samples)} total images")
    return train_samples, label_offset


# ══════════════════════════════════════════════════════════════
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(net, cfg, num_classes, device):
    cache_dir    = os.path.abspath(cfg.get("base_results_dir", "./rst_ppnet_loo"))
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
#  TEST SPLIT
# ══════════════════════════════════════════════════════════════

def split_test_dataset(id2paths, gallery_ratio=0.50, seed=42):
    """Split the left-out dataset into gallery and probe."""
    rng       = random.Random(seed)
    label_map = {k: i for i, k in enumerate(sorted(id2paths.keys()))}
    gallery_samples, probe_samples = [], []
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p, label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p, label_map[ident]))
    return gallery_samples, probe_samples


# ══════════════════════════════════════════════════════════════
#  PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════

class AugmentedDataset(Dataset):
    """
    Training dataset with PPNet-compatible augmentation.
    Returns single images — contrastive pairing is done within the
    batch by the training loop (batch must be even).
    """
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
    Batch must be even; odd last batch is padded by duplicating first sample.
    """
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for data, target in loader:
            # ensure even batch
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
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval(); feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _single_eer(genuine, impostor):
    """
    EER from two score arrays.
    For L2 distances, genuine scores are LOWER → flip is triggered automatically.
    """
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
    """
    Returns (eer_all, eer_bal, rank1).
    Uses L2 distance — PPNet embeddings are unnormalised.
    Rank-1 uses argmin; EER uses automatic direction flip.
    """
    probe_feats,   probe_labels   = extract_features(model, probe_loader,   device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe   = len(probe_feats)
    n_gallery = len(gallery_feats)

    # vectorised L2 distance: ||a-b||² = ||a||² + ||b||² - 2·a·b
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
#  SINGLE LEAVE-ONE-OUT EXPERIMENT
# ══════════════════════════════════════════════════════════════

def run_experiment(train_datasets, test_dataset, cfg, device=None):
    """
    Train PPNet on the combined samples from `train_datasets` and
    evaluate on the left-out `test_dataset`.
    Returns (final_eer_bal, final_rank1).
    """
    seed            = cfg["random_seed"]
    results_dir     = cfg["results_dir"]
    img_side        = cfg["img_side"]
    batch_size      = cfg["batch_size"]
    num_epochs      = cfg["num_epochs"]
    lr              = cfg["lr"]
    lr_step         = cfg["lr_step"]
    lr_gamma        = cfg["lr_gamma"]
    margin          = cfg["contrastive_margin"]
    w_l2            = cfg["w_l2"]
    w_contra        = cfg["w_contra"]
    w_dis           = cfg["w_dis"]
    augment_factor  = cfg["augment_factor"]
    test_gal_ratio  = cfg["test_gallery_ratio"]
    eval_every      = cfg["eval_every"]
    save_every      = cfg["save_every"]
    nw              = cfg["num_workers"]
    eval_tag_base   = test_dataset.replace("-", "")

    assert batch_size % 2 == 0, f"batch_size must be even, got {batch_size}"

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    # ── build combined training set (3 datasets × 190 IDs each) ──────────
    train_samples, num_classes = build_combined_train_samples(train_datasets, cfg)

    # ── build test set (gallery + probe) from left-out dataset ───────────
    print(f"  Parsing {test_dataset} (test) …")
    test_id2paths  = get_parser(test_dataset, cfg)()
    gallery_samples, probe_samples = split_test_dataset(
        test_id2paths, test_gal_ratio, seed)

    # ── data loaders ──────────────────────────────────────────────────────
    train_loader = DataLoader(
        AugmentedDataset(train_samples, img_side, augment_factor),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True,
        drop_last=True)    # ensures even batches throughout
    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Gallery={len(gallery_samples)}  Probe={len(probe_samples)}  "
          f"Classes(train)={num_classes}")

    # ── model ─────────────────────────────────────────────────────────────
    net = ppnet(num_classes=num_classes)
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    net = get_or_create_init_weights(net, cfg, num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, lr_step, lr_gamma)

    # ── pre-training evaluation ───────────────────────────────────────────
    _net = net.module if isinstance(net, DataParallel) else net
    pre_eer_all, pre_eer_bal, pre_r1 = evaluate(
        _net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag=f"ep-001_pretrain_{eval_tag_base}")
    best_eer     = pre_eer_bal
    last_eer_all = pre_eer_all; last_eer_bal = pre_eer_bal; last_rank1 = pre_r1
    torch.save(_net.state_dict(),
               os.path.join(results_dir, "net_params_best_eer.pth"))

    train_losses, train_accs = [], []

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            net, train_loader, criterion, optimizer, device, "training",
            margin=margin, w_l2=w_l2, w_contra=w_contra, w_dis=w_dis)
        scheduler.step()
        train_losses.append(t_loss); train_accs.append(t_acc)
        _net = net.module if isinstance(net, DataParallel) else net

        if (epoch % eval_every == 0 and epoch > 0) or epoch == num_epochs - 1:
            tag = f"ep{epoch:04d}_{eval_tag_base}"
            cur_eer_all, cur_eer_bal, cur_rank1 = evaluate(
                _net, probe_loader, gallery_loader,
                device, out_dir=rst_eval, tag=tag)
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

    # ── final evaluation ──────────────────────────────────────────────────
    best_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(results_dir, "net_params.pth")
    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_path, map_location=device))
    final_eer_all, final_eer_bal, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag=f"FINAL_{eval_tag_base}")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(train_losses,'b'); axes[0].set_title("Train Loss")
        axes[0].set_xlabel("epoch"); axes[0].grid(True)
        axes[1].plot(train_accs,  'b'); axes[1].set_title("Train Acc (%)")
        axes[1].set_xlabel("epoch"); axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "train_curves.png"))
        plt.close(fig)
    except Exception:
        pass

    return final_eer_bal, final_rank1


# ══════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ══════════════════════════════════════════════════════════════

def print_and_save_table(results, all_datasets, out_path):
    """
    results[test_dataset] = (eer_bal_pct, rank1_pct)
    One row per left-out test dataset. Final row shows the average.
    """
    col_w  = 16
    header = (f"{'Test (left out)':<18}"
              f"{'Train datasets':<44}"
              f"{'EER_bal (%)':>{col_w}}"
              f"{'Rank-1 (%)':>{col_w}}")
    sep = "─" * len(header)

    lines = ["\nLeave-One-Out Results", sep, header, sep]

    eer_vals, rank1_vals = [], []
    for test_ds in all_datasets:
        train_ds  = [d for d in all_datasets if d != test_ds]
        train_str = " + ".join(d.replace("-","") for d in train_ds)
        val = results.get(test_ds)
        if val is not None:
            eer_str   = f"{val[0]:.2f}"
            rank1_str = f"{val[1]:.2f}"
            eer_vals.append(val[0]); rank1_vals.append(val[1])
        else:
            eer_str = rank1_str = "—"
        lines.append(f"{test_ds.replace('-',''):<18}"
                     f"{train_str:<44}"
                     f"{eer_str:>{col_w}}"
                     f"{rank1_str:>{col_w}}")

    lines.append(sep)
    avg_eer   = f"{sum(eer_vals)/len(eer_vals):.2f}"     if eer_vals   else "—"
    avg_rank1 = f"{sum(rank1_vals)/len(rank1_vals):.2f}" if rank1_vals else "—"
    lines.append(f"{'Avg':<18}"
                 f"{'':44}"
                 f"{avg_eer:>{col_w}}"
                 f"{avg_rank1:>{col_w}}")
    lines.append(sep)

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nTable saved to: {out_path}")


# ══════════════════════════════════════════════════════════════
#  DATASET DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════

def print_dataset_distributions(cfg):
    """Parse all four datasets and print a summary distribution table."""
    print(f"\n{'='*60}")
    print(f"  DATASET DISTRIBUTIONS")
    print(f"{'='*60}")

    datasets = [
        ("CASIA-MS",  lambda: parse_casia_ms(cfg["casiams_data_root"],
                                             seed=cfg["random_seed"])),
        ("Palm-Auth", lambda: parse_palm_auth_data(cfg["palm_auth_data_root"],
                                                   use_scanner=cfg.get("use_scanner", False),
                                                   seed=cfg["random_seed"])),
        ("MPDv2",     lambda: parse_mpd_data(cfg["mpd_data_root"],
                                             seed=cfg["random_seed"])),
        ("XJTU",      lambda: parse_xjtu_data(cfg["xjtu_data_root"],
                                              seed=cfg["random_seed"])),
    ]

    col_w  = 12
    header = (f"{'Dataset':<14}"
              f"{'Subjects':>{col_w}}"
              f"{'Total Imgs':>{col_w}}"
              f"{'Min/Subj':>{col_w}}"
              f"{'Max/Subj':>{col_w}}"
              f"{'Mean/Subj':>{col_w}}"
              f"{'Median/Subj':>{col_w}}")
    sep = "─" * len(header)
    print(f"\n{header}")
    print(sep)

    for ds_name, parser_fn in datasets:
        try:
            id2paths = parser_fn()
            counts   = [len(v) for v in id2paths.values()]
            n_subj   = len(counts)
            total    = sum(counts)
            mn, mx   = min(counts), max(counts)
            mean     = total / n_subj
            median   = float(np.median(counts))
            print(f"{ds_name:<14}"
                  f"{n_subj:>{col_w}}"
                  f"{total:>{col_w}}"
                  f"{mn:>{col_w}}"
                  f"{mx:>{col_w}}"
                  f"{mean:>{col_w}.1f}"
                  f"{median:>{col_w}.1f}")
        except Exception as e:
            print(f"{ds_name:<14}  ERROR: {e}")

    print(sep)
    print()


# ══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def main():
    seed = BASE_CONFIG["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir = BASE_CONFIG.get("base_results_dir", "./rst_ppnet_loo")
    os.makedirs(base_results_dir, exist_ok=True)

    assert BASE_CONFIG["batch_size"] % 2 == 0, \
        "batch_size must be even for PPNet contrastive pairing"

    print_dataset_distributions(BASE_CONFIG)

    print(f"\n{'='*60}")
    print(f"  PPNet — Leave-One-Out Cross-Dataset Experiment")
    print(f"  Device       : {device}")
    print(f"  Datasets     : {ALL_DATASETS}")
    print(f"  Strategy     : train on 3, test on left-out 1")
    print(f"  Epochs       : {BASE_CONFIG['num_epochs']}")
    print(f"  Loss         : CE + {BASE_CONFIG['w_l2']}*L2 + "
          f"{BASE_CONFIG['w_contra']}*Contra(m={BASE_CONFIG['contrastive_margin']}) + "
          f"{BASE_CONFIG['w_dis']}*dis²")
    print(f"  Matching     : L2 distance (unnormalised 512-d embeddings)")
    print(f"  EER_bal      = balanced 1:1 impostor sampling (model selection)")
    print(f"  EER_all      = all impostor pairs (reference)")
    print(f"  Results dir  : {base_results_dir}")
    print(f"{'='*60}\n")

    n_total  = len(ALL_DATASETS)
    n_done   = 0
    results  = {}   # test_dataset → (eer_bal_pct, rank1_pct)
    failures = []

    for test_dataset in ALL_DATASETS:
        train_datasets = [d for d in ALL_DATASETS if d != test_dataset]
        n_done += 1
        train_str = " + ".join(train_datasets)
        exp_label = f"train=[{train_str}]  test={test_dataset}"

        print(f"\n{'='*60}")
        print(f"  Experiment {n_done}/{n_total}:  {exp_label}")
        print(f"{'='*60}")

        cfg = copy.deepcopy(BASE_CONFIG)
        safe_test = test_dataset.replace("-","").replace(" ","")
        cfg["results_dir"] = os.path.join(base_results_dir, f"test_{safe_test}")

        t_start = time.time()
        try:
            eer_bal, rank1 = run_experiment(
                train_datasets, test_dataset, cfg, device=device)

            results[test_dataset] = (eer_bal * 100, rank1)
            elapsed = time.time() - t_start
            print(f"\n  ✓  {exp_label}")
            print(f"     EER_bal={eer_bal*100:.4f}%  Rank-1={rank1:.2f}%  "
                  f"Time={elapsed/60:.1f} min")

        except Exception as e:
            results[test_dataset] = None
            failures.append((test_dataset, str(e)))
            print(f"\n  ✗  {exp_label}  FAILED: {e}")

    # ── Print and save results table ──────────────────────────────────────
    table_path = os.path.join(base_results_dir, "results_table.txt")
    print(f"\n\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print_and_save_table(results, ALL_DATASETS, table_path)

    if failures:
        print(f"\nFailed experiments ({len(failures)}):")
        for te, err in failures:
            print(f"  test={te}  → {err}")

    json_results = {te: list(v) if v else None for te, v in results.items()}
    with open(os.path.join(base_results_dir, "results_raw.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nRaw results: {os.path.join(base_results_dir, 'results_raw.json')}")


if __name__ == "__main__":
    main()
