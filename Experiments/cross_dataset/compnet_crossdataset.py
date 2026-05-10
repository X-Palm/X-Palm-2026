"""
CompNet — Full Cross-Dataset Experiment Runner
==================================================
Runs ALL combinations of train × test datasets and prints a
summary table of EER and Rank-1 at the end.

Train datasets : CASIA-MS | X-Palm | MPDv2 | XJTU
Test  datasets : CASIA-MS | X-Palm | MPDv2 | XJTU

Results are saved to:
  {BASE_RESULTS_DIR}/train_{X}_test_{Y}/   ← per-experiment outputs
  {BASE_RESULTS_DIR}/results_table.txt     ← final EER / Rank-1 table (with Avg column)
  {BASE_RESULTS_DIR}/results_raw.json      ← raw numbers as JSON
"""

# ==============================================================
#  EXPERIMENT GRID  — edit these two lists to change what runs
# ==============================================================
TRAIN_DATASETS = ["X-Palm", "CASIA-MS", "MPDv2", "XJTU"]
TEST_DATASETS  = ["X-Palm", "CASIA-MS", "MPDv2", "XJTU"]

# ==============================================================
#  BASE CONFIG  — shared across all experiments
# ==============================================================
BASE_CONFIG = {
    # ── Dataset paths ──────────────────────────────────────────
    "casiams_data_root"    : "./CASIA-MS-ROI",
    "x_palm_data_root"  : "./X_Palm_ROI",
    "mpd_data_root"        : "./MPDv2__roi",
    "xjtu_data_root"       : "./XJTU-UP",

    # ── Splitting ──────────────────────────────────────────────
    "train_subject_ratio"  : 0.80,
    "test_gallery_ratio"   : 0.50,

    # ── Palm-Auth toggle ───────────────────────────────────────
    "use_scanner"          : True,

    # ── Model ──────────────────────────────────────────────────
    "img_side"             : 128,
    "embedding_dim"        : 512,
    "dropout"              : 0.25,
    "arcface_s"            : 30.0,
    "arcface_m"            : 0.50,

    # ── Training ───────────────────────────────────────────────
    "batch_size"           : 128,
    "num_epochs"           : 200,
    "lr"                   : 0.001,
    "lr_step"              : 30,
    "lr_gamma"             : 0.8,
    "augment_factor"       : 2,

    # ── Misc ───────────────────────────────────────────────────
    "base_results_dir"     : "./rst_compnet_all",
    "random_seed"          : 42,
    "save_every"           : 50,
    "eval_every"           : 50,
    "num_workers"          : 4,
    "resume"               : False,
    "eval_only"            : False,
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
from torch.nn import Parameter, DataParallel
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
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

ALLOWED_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

N_HIGH = 160
N_LOW  = 40

TARGET_HIGH_CASIA = 33
TARGET_LOW_CASIA  = 15
TARGET_HIGH_MPD   = 33
TARGET_LOW_MPD    = 15
TARGET_HIGH_XJTU  = 33
TARGET_LOW_XJTU   = 15


XJTU_VARIATIONS = [
    ("iPhone", "Flash"),
    ("iPhone", "Nature"),
    ("huawei", "Flash"),
    ("huawei", "Nature"),
]


# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in = channel_in; self.channel_out = channel_out
        self.kernel_size = kernel_size; self.stride = stride
        self.padding = padding; self.init_ratio = max(init_ratio, 1e-6)
        self.kernel = 0
        _S = 9.2 * self.init_ratio; _F = 0.057 / self.init_ratio; _G = 2.0
        self.gamma = nn.Parameter(torch.FloatTensor([_G]))
        self.sigma = nn.Parameter(torch.FloatTensor([_S]))
        self.theta = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_F]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def _gen(self, ksize, c_in, c_out, sigma, gamma, theta, f, psi):
        half = ksize // 2; ksz = 2 * half + 1
        y0 = torch.arange(-half, half + 1).float()
        x0 = torch.arange(-half, half + 1).float()
        y  = y0.view(1,-1).repeat(c_out, c_in, ksz, 1)
        x  = x0.view(-1,1).repeat(c_out, c_in, 1, ksz)
        x  = x.to(sigma.device); y = y.to(sigma.device)
        xt =  x*torch.cos(theta.view(-1,1,1,1)) + y*torch.sin(theta.view(-1,1,1,1))
        yt = -x*torch.sin(theta.view(-1,1,1,1)) + y*torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(-0.5*((gamma*xt)**2+yt**2)/(8*sigma.view(-1,1,1,1)**2)
            ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt+psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        self.kernel = self._gen(self.kernel_size, self.channel_in,
                                self.channel_out, self.sigma, self.gamma,
                                self.theta, self.f, self.psi)
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor   = GaborConv2d(channel_in, n_competitor, ksize,
                                   stride, padding, init_ratio)
        self.a       = nn.Parameter(torch.FloatTensor([1]))
        self.b       = nn.Parameter(torch.FloatTensor([0]))
        self.argmax  = nn.Softmax(dim=1)
        self.conv1   = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor(x)
        x = self.argmax((x - self.b) * self.a)
        return self.conv2(self.maxpool(self.conv1(x)))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight      = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine) if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1 - one_hot) * cosine))
        return self.s * cosine


class CompNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=512,
                 arcface_s=30.0, arcface_m=0.50, dropout=0.25):
        super().__init__()
        self.cb1  = CompetitiveBlock(1, 9, 35, 3, 0, init_ratio=1.00)
        self.cb2  = CompetitiveBlock(1, 9, 17, 3, 0, init_ratio=0.50)
        self.cb3  = CompetitiveBlock(1, 9,  7, 3, 0, init_ratio=0.25)
        self.fc   = nn.Linear(9708, embedding_dim)
        self.drop = nn.Dropout(p=dropout)
        self.arc  = ArcMarginProduct(embedding_dim, num_classes,
                                     s=arcface_s, m=arcface_m)

    def _backbone(self, x):
        x1 = self.cb1(x).flatten(1); x2 = self.cb2(x).flatten(1)
        x3 = self.cb3(x).flatten(1)
        return self.fc(torch.cat([x1, x2, x3], dim=1))

    def forward(self, x, y=None):
        return self.arc(self.drop(self._backbone(x)), y)

    @torch.no_grad()
    def get_embedding(self, x):
        return F.normalize(self._backbone(x), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION
# ══════════════════════════════════════════════════════════════

class NormSingleROI:
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

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


def parse_palm_auth_data(data_root, use_scanner=False):
    IMG_EXTS = {".jpg",".jpeg",".bmp",".png"}
    id2paths  = defaultdict(list)
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
    result = dict(id2paths); counts = [len(v) for v in result.values()]
    mode = (f"perspective + scanner ({', '.join(sorted(ALLOWED_SPECTRA))})"
            if use_scanner else "perspective only")
    print(f"  [Palm-Auth/{mode}]")
    print(f"    ids={len(result)}  total={sum(counts)}  "
          f"per-id min/max/mean={min(counts)}/{max(counts)}/{sum(counts)/len(counts):.1f}")
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
    all_ids = list(id_dev.keys()); rng.shuffle(all_ids)
    all_ids.sort(key=lambda i: len(id_dev[i].get("h",[]))+len(id_dev[i].get("m",[])),
                 reverse=True)
    if len(all_ids) < N_HIGH:
        raise ValueError(f"MPDv2: need {N_HIGH} IDs, found {len(all_ids)}")
    high_ids  = all_ids[:N_HIGH]
    low_cands = [i for i in all_ids[N_HIGH:]
                 if len(id_dev[i].get("h",[]))+len(id_dev[i].get("m",[]))>=TARGET_LOW_MPD]
    if len(low_cands) < N_LOW:
        raise ValueError("MPDv2: not enough low-group IDs")
    low_ids = low_cands[:N_LOW]
    def _sample(ident, target):
        paths = id_dev[ident].get("h",[]) + id_dev[ident].get("m",[])
        return rng.sample(paths, min(target, len(paths)))
    id2paths = {}
    for ident in high_ids: id2paths[ident] = _sample(ident, TARGET_HIGH_MPD)
    for ident in low_ids:  id2paths[ident] = _sample(ident, TARGET_LOW_MPD)
    actual = sum(len(v) for v in id2paths.values())
    hc = [len(id2paths[i]) for i in high_ids]; lc = [len(id2paths[i]) for i in low_ids]
    cutoff_h = len(id_dev[high_ids[-1]].get("h",[]))+len(id_dev[high_ids[-1]].get("m",[]))
    cutoff_l = len(id_dev[low_ids[-1]].get("h",[]))+len(id_dev[low_ids[-1]].get("m",[]))
    print(f"  [MPDv2] ids={len(id2paths)}  total={actual}")
    print(f"    High ({N_HIGH}×~{TARGET_HIGH_MPD}): min={min(hc)} max={max(hc)} mean={sum(hc)/N_HIGH:.1f} cutoff={cutoff_h}")
    print(f"    Low  ({N_LOW}×~{TARGET_LOW_MPD}):  min={min(lc)} max={max(lc)} mean={sum(lc)/N_LOW:.1f} cutoff={cutoff_l}")
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
                                            use_scanner=cfg.get("use_scanner", False))
    elif name == "mpdv2":
        return lambda: parse_mpd_data(cfg["mpd_data_root"], seed=seed)
    elif name == "xjtu":
        return lambda: parse_xjtu_data(cfg["xjtu_data_root"], seed=seed)
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'")


def _ds_key(name):
    return name.strip().lower().replace("-","").replace("_","")


# ══════════════════════════════════════════════════════════════
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(net, cfg, num_classes, device):
    """Per-model, per-num_classes weight cache to guarantee identical init.
    Cache is stored in base_results_dir."""
    cache_dir    = os.path.abspath(cfg.get("base_results_dir", "./rst_compnet_all"))
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
#  SPLITS
# ══════════════════════════════════════════════════════════════

def split_same_dataset(id2paths, train_subject_ratio=0.80,
                       gallery_ratio=0.50, seed=42):
    rng        = random.Random(seed)
    identities = sorted(id2paths.keys()); rng.shuffle(identities)
    n_train    = max(1, int(len(identities) * train_subject_ratio))
    train_ids  = identities[:n_train]; test_ids = identities[n_train:]
    train_label_map = {k: i for i, k in enumerate(train_ids)}
    test_label_map  = {k: i for i, k in enumerate(test_ids)}
    train_samples   = [(p, train_label_map[ident])
                       for ident in train_ids for p in id2paths[ident]]
    gallery_samples, probe_samples = [], []
    for ident in test_ids:
        paths = list(id2paths[ident]); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p, test_label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p, test_label_map[ident]))
    return train_samples, gallery_samples, probe_samples, train_label_map, test_label_map


def split_cross_dataset_test(id2paths, gallery_ratio=0.50, seed=42):
    rng       = random.Random(seed)
    label_map = {k: i for i, k in enumerate(sorted(id2paths.keys()))}
    gallery_samples, probe_samples = [], []
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p, label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p, label_map[ident]))
    return gallery_samples, probe_samples, label_map


# ══════════════════════════════════════════════════════════════
#  PYTORCH DATASETS
# ══════════════════════════════════════════════════════════════

class SingleDataset(Dataset):
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = T.Compose([T.Resize(img_side), T.ToTensor(),
                                    NormSingleROI(outchannels=1)])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


class AugmentedDataset(Dataset):
    def __init__(self, samples, img_side=128, augment_factor=1):
        self.samples = samples; self.augment_factor = augment_factor
        self.aug_transform = T.Compose([
            T.Resize(img_side),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.0,1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1),
                T.RandomChoice([
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.5*img_side, 0.0)),
                    T.RandomRotation(10, interpolation=Image.BICUBIC,
                                     expand=False, center=(0.0, 0.5*img_side)),
                ]),
            ]),
            T.ToTensor(), NormSingleROI(outchannels=1),
        ])
    def __len__(self): return len(self.samples) * self.augment_factor
    def __getitem__(self, index):
        real_idx = index % len(self.samples)
        path, label = self.samples[real_idx]
        return self.aug_transform(Image.open(path).convert("L")), label


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, optimizer, device, phase):
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if is_train: optimizer.zero_grad()
            output = model(data, target if is_train else None)
            loss   = criterion(output, target)
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


def compute_eer(scores_array):
    ins  = scores_array[scores_array[:, 1] ==  1, 0]
    outs = scores_array[scores_array[:, 1] == -1, 0]
    if len(ins) == 0 or len(outs) == 0: return 1.0, 0.0
    y   = np.concatenate([np.ones(len(ins)), np.zeros(len(outs))])
    s   = np.concatenate([ins, outs])
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    eer    = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = float(interp1d(fpr, thresholds)(eer))
    return eer, thresh


def evaluate(model, probe_loader, gallery_loader, device,
             out_dir=".", tag="eval"):
    probe_feats,   probe_labels   = extract_features(model, probe_loader,   device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe    = len(probe_feats)
    sim_matrix = probe_feats @ gallery_feats.T

    scores_list, labels_list = [], []
    for i in range(n_probe):
        for j in range(sim_matrix.shape[1]):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _     = compute_eer(scores_arr)

    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]] for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")

    print(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  SINGLE EXPERIMENT
# ══════════════════════════════════════════════════════════════

def run_experiment(train_data, test_data, cfg, device=None):
    """
    Train CompNet on `train_data` and evaluate on `test_data`.
    Returns (final_eer, final_rank1).
    """
    seed            = cfg["random_seed"]
    results_dir     = cfg["results_dir"]
    img_side        = cfg["img_side"]
    batch_size      = cfg["batch_size"]
    num_epochs      = cfg["num_epochs"]
    lr              = cfg["lr"]
    lr_step         = cfg["lr_step"]
    lr_gamma        = cfg["lr_gamma"]
    dropout         = cfg["dropout"]
    arcface_s       = cfg["arcface_s"]
    arcface_m       = cfg["arcface_m"]
    embedding_dim   = cfg["embedding_dim"]
    augment_factor  = cfg["augment_factor"]
    test_gal_ratio  = cfg["test_gallery_ratio"]
    train_sub_ratio = cfg["train_subject_ratio"]
    eval_every      = cfg["eval_every"]
    save_every      = cfg["save_every"]
    nw              = cfg["num_workers"]

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    same_dataset  = (_ds_key(train_data) == _ds_key(test_data))
    eval_tag_base = test_data.replace("-","")

    # ── select training IDs ───────────────────────────────────────────────
    if same_dataset:
        print(f"  Parsing {train_data} (shared train+test) …")
        all_id2paths = get_parser(train_data, cfg)()
        (train_samples, gallery_samples, probe_samples,
         train_label_map, _) = split_same_dataset(
            all_id2paths, train_sub_ratio, test_gal_ratio, seed)
        num_classes   = len(train_label_map)

    else:
        print(f"  Parsing {train_data} (train) …")
        train_id2paths  = get_parser(train_data, cfg)()
        train_label_map = {k: i for i, k in enumerate(sorted(train_id2paths))}
        train_samples   = [(p, train_label_map[ident])
                           for ident, paths in train_id2paths.items()
                           for p in paths]
        num_classes = len(train_label_map)
        print(f"  Parsing {test_data} (test) …")
        test_id2paths   = get_parser(test_data, cfg)()
        gallery_samples, probe_samples, _ = split_cross_dataset_test(
            test_id2paths, test_gal_ratio, seed)

    # ── data loaders ──────────────────────────────────────────────────────
    train_loader = DataLoader(
        AugmentedDataset(train_samples, img_side, augment_factor),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Gallery={len(gallery_samples)}  Probe={len(probe_samples)}  Classes={num_classes}")

    # ── model ─────────────────────────────────────────────────────────────
    net = CompNet(num_classes, embedding_dim=embedding_dim,
                  arcface_s=arcface_s, arcface_m=arcface_m, dropout=dropout)
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    # Always load fixed init weights (ensures identical starting point)
    net = get_or_create_init_weights(net, cfg, num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, lr_step, lr_gamma)

    # ── pre-training evaluation ───────────────────────────────────────────
    _net = net.module if isinstance(net, DataParallel) else net
    pre_eer, pre_r1 = evaluate(_net, probe_loader, gallery_loader,
                                device, out_dir=rst_eval,
                                tag=f"ep-001_pretrain_{eval_tag_base}")
    best_eer = pre_eer; last_eer = pre_eer; last_rank1 = pre_r1
    torch.save(_net.state_dict(),
               os.path.join(results_dir, "net_params_best_eer.pth"))

    train_losses, train_accs = [], []

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            net, train_loader, criterion, optimizer, device, "training")
        scheduler.step()
        train_losses.append(t_loss); train_accs.append(t_acc)
        _net = net.module if isinstance(net, DataParallel) else net

        if (epoch % eval_every == 0 and epoch > 0) or epoch == num_epochs - 1:
            tag = f"ep{epoch:04d}_{eval_tag_base}"
            cur_eer, cur_rank1 = evaluate(_net, probe_loader, gallery_loader,
                                           device, out_dir=rst_eval, tag=tag)
            last_eer, last_rank1 = cur_eer, cur_rank1
            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save(_net.state_dict(),
                           os.path.join(results_dir, "net_params_best_eer.pth"))
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ts        = time.strftime("%H:%M:%S")
            eer_str   = f"{last_eer*100:.4f}%"  if not math.isnan(last_eer)   else "N/A"
            rank1_str = f"{last_rank1:.2f}%"     if not math.isnan(last_rank1) else "N/A"
            print(f"  [{ts}] ep {epoch:04d} | loss={t_loss:.4f} | "
                  f"EER={eer_str}  Rank-1={rank1_str}")

        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))

    # ── final evaluation (best checkpoint) ────────────────────────────────
    best_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(results_dir, "net_params.pth")
    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_path, map_location=device))
    final_eer, final_rank1 = evaluate(
        eval_net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag=f"FINAL_{eval_tag_base}")

    # ── save train curves ─────────────────────────────────────────────────
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

    return final_eer, final_rank1


# ══════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ══════════════════════════════════════════════════════════════

def print_and_save_table(results, train_datasets, test_datasets, out_path):
    """
    results[(train, test)] = (eer_pct, rank1_pct)
    Prints two tables (EER and Rank-1) with an Avg column and saves to out_path.
    """
    col_w    = 14
    td_label = [t.replace("-","") for t in test_datasets] + ["Avg"]
    header   = f"{'Train\\Test':<14}" + "".join(f"{t:>{col_w}}" for t in td_label)
    sep      = "─" * len(header)

    lines = []
    for metric_label, idx in [("EER (%)", 0), ("Rank-1 (%)", 1)]:
        lines.append(f"\n{metric_label} Results")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        for tr in train_datasets:
            row  = f"{tr.replace('-',''):<14}"
            vals = []
            for te in test_datasets:
                val  = results.get((tr, te))
                cell = f"{val[idx]:.2f}" if val is not None else "—"
                row += f"{cell:>{col_w}}"
                if val is not None:
                    vals.append(val[idx])
            avg_cell = f"{sum(vals)/len(vals):.2f}" if vals else "—"
            row += f"{avg_cell:>{col_w}}"
            lines.append(row)
        lines.append(sep)

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nTable saved to: {out_path}")



def print_dataset_distributions(cfg):
    """Parse all four datasets and print a summary distribution table."""
    print(f"\n{'='*60}")
    print(f"  DATASET DISTRIBUTIONS")
    print(f"{'='*60}")

    datasets = [
        ("CASIA-MS",  lambda: parse_casia_ms(cfg["casiams_data_root"],
                                             seed=cfg["random_seed"])),
        ("Palm-Auth", lambda: parse_palm_auth_data(cfg["palm_auth_data_root"],
                                                   use_scanner=cfg.get("use_scanner", False))),
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir = BASE_CONFIG.get("base_results_dir", "./rst_compnet_all")
    os.makedirs(base_results_dir, exist_ok=True)

    print_dataset_distributions(BASE_CONFIG)

    print(f"\n{'='*60}")
    print(f"  CompNet — Full Cross-Dataset Experiment")
    print(f"  Device      : {device}")
    print(f"  Train sets  : {TRAIN_DATASETS}")
    print(f"  Test  sets  : {TEST_DATASETS}")
    print(f"  Epochs      : {BASE_CONFIG['num_epochs']}")
    print(f"  Results dir : {base_results_dir}")
    print(f"{'='*60}\n")

    # ── Loop over all combinations ────────────────────────────────────────
    n_total  = len(TRAIN_DATASETS) * len(TEST_DATASETS)
    n_done   = 0
    results  = {}   # (train, test) → (eer_pct, rank1_pct)
    failures = []

    for train_data in TRAIN_DATASETS:
        for test_data in TEST_DATASETS:
            n_done += 1
            exp_label = f"train={train_data}  test={test_data}"
            print(f"\n{'='*60}")
            print(f"  Experiment {n_done}/{n_total}:  {exp_label}")
            print(f"{'='*60}")

            # ── build per-experiment config ────────────────────────────────
            cfg = copy.deepcopy(BASE_CONFIG)
            cfg["train_data"] = train_data
            cfg["test_data"]  = test_data

            # Dedicated results folder per experiment
            safe_train = train_data.replace("-","").replace(" ","")
            safe_test  = test_data.replace("-","").replace(" ","")
            cfg["results_dir"] = os.path.join(
                base_results_dir, f"train_{safe_train}_test_{safe_test}")

            t_start = time.time()
            try:
                eer, rank1 = run_experiment(
                    train_data, test_data, cfg, device=device)

                results[(train_data, test_data)] = (eer * 100, rank1)
                elapsed = time.time() - t_start
                print(f"\n  ✓  {exp_label}")
                print(f"     EER={eer*100:.4f}%  Rank-1={rank1:.2f}%  "
                      f"Time={elapsed/60:.1f} min")

            except Exception as e:
                results[(train_data, test_data)] = None
                failures.append((train_data, test_data, str(e)))
                print(f"\n  ✗  {exp_label}  FAILED: {e}")

    # ── Print and save results table ──────────────────────────────────────
    table_path = os.path.join(base_results_dir, "results_table.txt")
    print(f"\n\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print_and_save_table(results, TRAIN_DATASETS, TEST_DATASETS, table_path)

    if failures:
        print(f"\nFailed experiments ({len(failures)}):")
        for tr, te, err in failures:
            print(f"  train={tr}  test={te}  → {err}")

    # Also save raw numbers as JSON for later analysis
    json_results = {f"{tr}→{te}": list(v) if v else None
                    for (tr, te), v in results.items()}
    with open(os.path.join(base_results_dir, "results_raw.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nRaw results saved to: "
          f"{os.path.join(base_results_dir, 'results_raw.json')}")


if __name__ == "__main__":
    main()
