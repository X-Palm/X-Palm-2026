"""
PalmBridge — Full Cross-Dataset Experiment Runner
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
#  EXPERIMENT GRID
# ==============================================================
TRAIN_DATASETS = ["X-Palm", "CASIA-MS", "MPDv2", "XJTU"]
TEST_DATASETS  = ["X-Palm", "CASIA-MS", "MPDv2", "XJTU"]

# ==============================================================
#  BASE CONFIG
# ==============================================================
BASE_CONFIG = {
    "casiams_data_root"    : "./CASIA-MS-ROI",
    "x_palm_data_root"  : "./X_Palm_ROI",
    "mpd_data_root"        : "./MPDv2__roi",
    "xjtu_data_root"       : "./XJTU-UP",

    "train_subject_ratio"  : 0.80,
    "test_gallery_ratio"   : 0.50,
    "use_scanner"          : True,

    "img_side"             : 128,
    "embedding_dim"        : 512,
    "dropout"              : 0.25,

    "batch_size"           : 16,
    "num_epochs"           : 200,
    "lr"                   : 1e-3,
    "weight_decay"         : 5e-4,

    "base_results_dir"     : "./rst_palmbridge_all",
    "random_seed"          : 42,
    "eval_every"           : 5,
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

# ── PalmBridge hyperparameters ────────────────────────────────
FEATURE_DIM       = 512
NUM_PB_VECTORS    = 512
NUM_GABOR_FILTERS = 32
GABOR_KERNEL_SIZE = 15
W_MAP             = 0.3
W_ORI             = 0.7
ALPHA             = 0.1
BETA              = 1.0
LAMBDA_CON        = 0.25
ARC_S             = 48.0
ARC_M             = 0.40
WARMUP_EPOCHS     = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
#  MODEL  (unchanged)
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

    def _build_filters(self):
        theta = self.theta
        sigma = self.sigma.abs().clamp(min=0.5)
        lambd = self.lambd.abs().clamp(min=1.0)
        psi   = self.psi
        gamma = self.gamma.abs().clamp(min=0.1)
        xx = self.xx.unsqueeze(0); yy = self.yy.unsqueeze(0)
        cos_t = torch.cos(theta).view(-1,1,1); sin_t = torch.sin(theta).view(-1,1,1)
        sigma = sigma.view(-1,1,1); lambd = lambd.view(-1,1,1)
        psi   = psi.view(-1,1,1);   gamma = gamma.view(-1,1,1)
        x_rot =  xx * cos_t + yy * sin_t
        y_rot = -xx * sin_t + yy * cos_t
        envelope = torch.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2.0 * sigma**2))
        kernel   = envelope * torch.cos(2.0 * math.pi * x_rot / lambd + psi)
        kernel   = kernel - kernel.mean(dim=(1,2), keepdim=True)
        return kernel.unsqueeze(1).contiguous()

    def forward(self, x):
        return F.conv2d(x, self._build_filters(), padding=self.ks // 2)


class CompetitivePool(nn.Module):
    def forward(self, x):
        out, _ = x.abs().max(dim=1, keepdim=True)
        return out


class CompNet(nn.Module):
    def __init__(self, num_classes):
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
        self.block4 = _block(128, 256); self.gap   = nn.AdaptiveAvgPool2d((4,4))
        self.embed  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, FEATURE_DIM, bias=False),
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

    def forward(self, x):
        g = F.relu(self.gabor(x))
        g = self.gbn(self.compete(g))
        f = self.pool1(self.block1(g))
        f = self.pool2(self.block2(f))
        f = self.pool3(self.block3(f))
        f = self.gap(self.block4(f))
        return F.normalize(self.embed(f), p=2, dim=1)


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
        return self.P[idx], idx

    def forward(self, z):
        z_tilde, indices = self._nearest_vector(z)
        z_hat            = W_ORI * z + W_MAP * z_tilde
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
        return {"mean_cosine": off.mean().item(),
                "near_duplicate": (off > 0.9).float().mean().item()}


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.s     = ARC_S
        self.cos_m = math.cos(ARC_M); self.sin_m = math.sin(ARC_M)
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
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1,1), 1.0)
        logits  = self.s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta)
        return self.ce(logits, labels)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION  (unchanged)
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

def get_train_transform(img_side):
    return transforms.Compose([
        transforms.Resize(img_side),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            transforms.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.0,1.0)),
            transforms.RandomPerspective(distortion_scale=0.15, p=1),
            transforms.RandomChoice([
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(0.5*img_side, 0.0)),
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(0.0, 0.5*img_side)),
            ]),
        ]),
        transforms.ToTensor(), NormSingleROI(outchannels=1),
    ])


def get_eval_transform(img_side):
    return transforms.Compose([
        transforms.Resize(img_side),
        transforms.ToTensor(),
        NormSingleROI(outchannels=1),
    ])


class TrainDataset(Dataset):
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = get_train_transform(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


class EvalDataset(Dataset):
    def __init__(self, samples, img_side=128):
        self.samples   = samples
        self.transform = get_eval_transform(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), label


def make_loader(samples, train=False, batch_size=16, img_side=128, num_workers=4):
    ds = TrainDataset(samples, img_side) if train else EvalDataset(samples, img_side)
    return DataLoader(ds, batch_size=min(batch_size, len(samples)),
                      shuffle=train, num_workers=num_workers,
                      pin_memory=(DEVICE == "cuda"),
                      drop_last=train and len(samples) > batch_size)


# ══════════════════════════════════════════════════════════════
#  DATASET PARSERS  (from CompNet cross-dataset)
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
    selected = sorted(rng.sample(all_ids, N_HIGH + N_LOW)); rng.shuffle(selected)
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
    print(f"  [CASIA-MS] ids={len(id2paths)}  total={actual}")
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
    print(f"  [Palm-Auth] ids={len(result)}  total={sum(counts)}")
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
    print(f"  [MPDv2] ids={len(id2paths)}  total={actual}")
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
    if len(all_ids) < N_HIGH + N_LOW:
        raise ValueError(f"XJTU: need {N_HIGH+N_LOW} IDs, found {len(all_ids)}")
    selected = sorted(rng.sample(all_ids, N_HIGH + N_LOW)); rng.shuffle(selected)
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
    print(f"  [XJTU] ids={len(id2paths)}  total={actual}")
    return id2paths


def get_parser(dataset_name, cfg):
    name = _ds_key(dataset_name)
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
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(backbone, palmbridge, arcface, num_classes,
                               cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    weights_path = os.path.join(cache_dir,
                                f"init_weights_PalmBridge_nc{num_classes}.pth")
    if os.path.exists(weights_path):
        print(f"  Loading cached init weights: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        palmbridge.load_state_dict(ckpt["palmbridge"])
        arcface.load_state_dict(ckpt["arcface"])
    else:
        print(f"  Saving init weights: {weights_path}")
        torch.save({"backbone":   backbone.state_dict(),
                    "palmbridge": palmbridge.state_dict(),
                    "arcface":    arcface.state_dict()}, weights_path)


# ══════════════════════════════════════════════════════════════
#  TRAINING  (unchanged)
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
        totals["loss"] += loss.item(); totals["bak"] += L_bak.item()
        totals["con"]  += L_con.item(); totals["orth"] += L_o.item()

    nb = max(len(loader), 1)
    metrics = {k: v / nb for k, v in totals.items()}
    metrics["train_acc"] = n_correct / max(n_total, 1)
    return metrics


def build_optimizer(backbone, palmbridge, arcface, lr, weight_decay):
    params = (list(backbone.parameters()) +
              list(palmbridge.parameters()) +
              list(arcface.parameters()))
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs):
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return 0.1 + 0.9 * (epoch / max(WARMUP_EPOCHS - 1, 1))
        progress = (epoch - WARMUP_EPOCHS) / max(epochs - WARMUP_EPOCHS, 1)
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════
#  EVALUATION  (unchanged)
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
    eer, _     = compute_eer(scores_arr)

    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]] for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")

    pb_str = "PB" if apply_palmbridge else "Naive"
    print(f"  [{tag}|{pb_str}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  SINGLE EXPERIMENT
# ══════════════════════════════════════════════════════════════

def run_experiment(train_data, test_data, cfg, device=None):
    seed            = cfg["random_seed"]
    results_dir     = cfg["results_dir"]
    img_side        = cfg["img_side"]
    batch_size      = cfg["batch_size"]
    num_epochs      = cfg["num_epochs"]
    lr              = cfg["lr"]
    weight_decay    = cfg["weight_decay"]
    test_gal_ratio  = cfg["test_gallery_ratio"]
    train_sub_ratio = cfg["train_subject_ratio"]
    eval_every      = cfg["eval_every"]
    nw              = cfg["num_workers"]
    cache_dir       = cfg["base_results_dir"]

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    same_dataset  = (_ds_key(train_data) == _ds_key(test_data))
    eval_tag_base = test_data.replace("-","")

    if same_dataset:
        print(f"  Parsing {train_data} (shared train+test) …")
        all_id2paths = get_parser(train_data, cfg)()
        (train_samples, gallery_samples, probe_samples,
         train_label_map, _) = split_same_dataset(
            all_id2paths, train_sub_ratio, test_gal_ratio, seed)
        num_classes = len(train_label_map)
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

    train_loader   = make_loader(train_samples,   train=True,
                                 batch_size=batch_size, img_side=img_side, num_workers=nw)
    gallery_loader = make_loader(gallery_samples, train=False,
                                 batch_size=batch_size, img_side=img_side, num_workers=nw)
    probe_loader   = make_loader(probe_samples,   train=False,
                                 batch_size=batch_size, img_side=img_side, num_workers=nw)

    print(f"  Gallery={len(gallery_samples)}  Probe={len(probe_samples)}  Classes={num_classes}")

    backbone   = CompNet(num_classes).to(device)
    palmbridge = PalmBridge().to(device)
    arcface    = ArcFaceLoss(num_classes).to(device)

    get_or_create_init_weights(backbone, palmbridge, arcface,
                               num_classes, cache_dir, device)

    optimizer = build_optimizer(backbone, palmbridge, arcface, lr, weight_decay)
    scheduler = build_scheduler(optimizer, num_epochs)
    best_eer  = float("inf")
    ckpt_path = os.path.join(results_dir, "net_params_best_eer.pt")

    # Pre-training baseline
    pre_eer, _ = evaluate(backbone, palmbridge, probe_loader, gallery_loader,
                          rst_eval, f"ep-001_pretrain_{eval_tag_base}",
                          apply_palmbridge=False)
    best_eer = pre_eer

    for epoch in range(1, num_epochs + 1):
        m = train_one_epoch(backbone, palmbridge, arcface,
                            train_loader, optimizer, epoch)
        scheduler.step()

        phase = "WARMUP" if epoch <= WARMUP_EPOCHS else "PalmBridge"
        print(f"  ep {epoch:03d}/{num_epochs} [{phase}]  "
              f"loss={m['loss']:.4f}  bak={m['bak']:.4f}  "
              f"con={m['con']:.4f}  orth={m['orth']:.4f}  "
              f"acc={m['train_acc']*100:.2f}%")

        if epoch % eval_every == 0 or epoch == num_epochs:
            use_pb = (epoch > WARMUP_EPOCHS)
            cur_eer, _ = evaluate(
                backbone, palmbridge, probe_loader, gallery_loader,
                rst_eval, f"ep{epoch:04d}_{eval_tag_base}",
                apply_palmbridge=use_pb)
            if cur_eer < best_eer:
                best_eer = cur_eer
                torch.save({"epoch": epoch,
                            "backbone":   backbone.state_dict(),
                            "palmbridge": palmbridge.state_dict(),
                            "arcface":    arcface.state_dict()}, ckpt_path)
                print(f"  *** New best EER: {best_eer*100:.4f}% ***")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        palmbridge.load_state_dict(ckpt["palmbridge"])

    print("  -- Naive baseline --")
    evaluate(backbone, palmbridge, probe_loader, gallery_loader,
             rst_eval, f"FINAL_naive_{eval_tag_base}", apply_palmbridge=False)
    print("  -- PalmBridge --")
    final_eer, final_rank1 = evaluate(
        backbone, palmbridge, probe_loader, gallery_loader,
        rst_eval, f"FINAL_pb_{eval_tag_base}", apply_palmbridge=True)

    return final_eer, final_rank1


# ══════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ══════════════════════════════════════════════════════════════

def print_and_save_table(results, train_datasets, test_datasets, out_path):
    col_w    = 14
    td_label = [t.replace("-","") for t in test_datasets] + ["Avg"]
    header   = f"{'Train\\Test':<14}" + "".join(f"{t:>{col_w}}" for t in td_label)
    sep      = "─" * len(header)
    lines    = []
    for metric_label, idx in [("EER (%)", 0), ("Rank-1 (%)", 1)]:
        lines.append(f"\n{metric_label} Results")
        lines.append(sep); lines.append(header); lines.append(sep)
        for tr in train_datasets:
            row = f"{tr.replace('-',''):<14}"; vals = []
            for te in test_datasets:
                val  = results.get((tr, te))
                cell = f"{val[idx]:.2f}" if val is not None else "—"
                row += f"{cell:>{col_w}}"
                if val is not None: vals.append(val[idx])
            avg_cell = f"{sum(vals)/len(vals):.2f}" if vals else "—"
            row += f"{avg_cell:>{col_w}}"
            lines.append(row)
        lines.append(sep)
    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f: f.write(text + "\n")
    print(f"\nTable saved to: {out_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    seed = BASE_CONFIG["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir = BASE_CONFIG["base_results_dir"]
    os.makedirs(base_results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PalmBridge — Full Cross-Dataset Experiment")
    print(f"  Device      : {device}")
    print(f"  Train sets  : {TRAIN_DATASETS}")
    print(f"  Test  sets  : {TEST_DATASETS}")
    print(f"  Epochs      : {BASE_CONFIG['num_epochs']}")
    print(f"  Results dir : {base_results_dir}")
    print(f"{'='*60}\n")

    n_total  = len(TRAIN_DATASETS) * len(TEST_DATASETS)
    n_done   = 0
    results  = {}
    failures = []

    for train_data in TRAIN_DATASETS:
        for test_data in TEST_DATASETS:
            n_done += 1
            exp_label = f"train={train_data}  test={test_data}"
            print(f"\n{'='*60}")
            print(f"  Experiment {n_done}/{n_total}:  {exp_label}")
            print(f"{'='*60}")

            cfg = copy.deepcopy(BASE_CONFIG)
            cfg["train_data"] = train_data
            cfg["test_data"]  = test_data
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

    table_path = os.path.join(base_results_dir, "results_table.txt")
    print(f"\n\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print_and_save_table(results, TRAIN_DATASETS, TEST_DATASETS, table_path)

    if failures:
        print(f"\nFailed experiments ({len(failures)}):")
        for tr, te, err in failures:
            print(f"  train={tr}  test={te}  → {err}")

    json_results = {f"{tr}→{te}": list(v) if v else None
                    for (tr, te), v in results.items()}
    with open(os.path.join(base_results_dir, "results_raw.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nRaw results saved to: "
          f"{os.path.join(base_results_dir, 'results_raw.json')}")


if __name__ == "__main__":
    main()
