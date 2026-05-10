"""
CCNet — Full Cross-Dataset Experiment Runner
==================================================
Runs ALL combinations of train × test datasets and prints a
summary table of EER and Rank-1 at the end.

Train datasets : CASIA-MS | X-Palm | MPDv2 | XJTU
Test  datasets : CASIA-MS | X-Palm | MPDv2 | XJTU

Model architecture and training: unchanged from official CCNet.
  - CompetitiveBlock_Mul_Ord_Comp (multi-order Gabor + spatial/channel
    competition + SE layer)
  - FC: 13152 → 4096 → 2048 + ArcFace (s=30, m=0.5)
  - Loss: ce_weight * CrossEntropy + con_weight * SupConLoss
  - Paired-image training (same-class pairs for SupConLoss)
  - comp_weight controls channel vs. spatial competition

Evaluation framework: follows CO3Net cross-dataset structure.
  - Same four dataset parsers with two-group sampling
  - Fixed init weights cache (per model class + num_classes)
  - EER_all (all impostor pairs) + EER_bal (balanced 1:1, 10 trials)
  - Results table saved as .txt and .json (with Avg column per row)

Note: final evaluation uses the last-epoch checkpoint (net_params.pth).

Results are saved to:
  {BASE_RESULTS_DIR}/train_{X}_test_{Y}/   ← per-experiment outputs
  {BASE_RESULTS_DIR}/results_table.txt     ← final EER_bal / Rank-1 table
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
    "casiams_data_root"    : "path to CASIA-MS-ROI",
    "x_palm_data_root"     : "path to X-Palm data",
    "mpd_data_root"        : "path to MPDv2 ROIs",
    "xjtu_data_root"       : "path to XJTU-UP",

    # ── Splitting ──────────────────────────────────────────────
    "train_subject_ratio"  : 0.80,
    "test_gallery_ratio"   : 0.50,

    # ── X-Palm toggle (include scanner images in training) ─────
    "use_scanner"          : True,

    # ── Model (official CCNet values) ──────────────────────────
    "img_side"             : 128,
    "comp_weight"          : 0.8,    # channel vs. spatial competition weight
    "dropout"              : 0.5,
    "arcface_s"            : 20.0,
    "arcface_m"            : 0.30,

    # ── Loss ───────────────────────────────────────────────────
    "ce_weight"            : 0.8,
    "con_weight"           : 0.2,
    "temperature"          : 0.07,

    # ── Training ───────────────────────────────────────────────
    "batch_size"           : 256,
    "num_epochs"           : 200,
    "lr"                   : 0.001,
    "lr_step"              : 17,     # proportional to official 500/3000 × 100
    "lr_gamma"             : 0.8,
    "augment_factor"       : 2,      # dataset expansion factor

    # ── Misc ───────────────────────────────────────────────────
    "base_results_dir"     : "./rst_ccnet_all",
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

# Spectral bands for the X-Palm multispectral scanner
ALLOWED_SPECTRA = {"green", "ir", "yellow", "red", "blue", "white"}

# Sampling parameters for CASIA-MS, MPDv2, and XJTU
# High-group: 160 IDs with ~33 images each
# Low-group:  40 IDs with ~15 images each
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
#  SUPCONLOSS  (exact copy from official CCNet/loss.py)
# ══════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature      = temperature
        self.contrast_mode    = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]; anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature; anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits        = torch.exp(logits) * logits_mask
        log_prob          = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


# ══════════════════════════════════════════════════════════════
#  CCNET ARCHITECTURE  (exact copy from official CCNet)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = init_ratio if init_ratio > 0 else 1.0
        self.kernel      = 0
        _SIGMA = 9.2   * self.init_ratio
        _FREQ  = 0.057 / self.init_ratio
        _GAMMA = 2.0
        self.gamma = nn.Parameter(torch.FloatTensor([_GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([_SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([_FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]),     requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out,
                     sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2; xmin = -xmax; ksize = xmax - xmin + 1
        y_0  = torch.arange(xmin, xmax + 1).float()
        x_0  = torch.arange(xmin, xmax + 1).float()
        y = y_0.view(1,-1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1,1).repeat(channel_out, channel_in, 1, ksize)
        x = x.float().to(sigma.device); y = y.float().to(sigma.device)
        xt =  x*torch.cos(theta.view(-1,1,1,1)) + y*torch.sin(theta.view(-1,1,1,1))
        yt = -x*torch.sin(theta.view(-1,1,1,1)) + y*torch.cos(theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5*((gamma*xt)**2 + yt**2) / (8*sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt + psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in,
                                   self.channel_out, self.sigma, self.gamma,
                                   self.theta, self.f, self.psi)
        self.kernel = kernel
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    """
    Multi-Order Comprehensive Competition Block (CCNet).
    1st order: LGC → spatial+channel competition → SE → conv → pool
    2nd order: LGC(on 1st Gabor output) → competition → SE → conv → pool
    Output: concatenation of 1st and 2nd order flattened features.
    """
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 weight, init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor_conv2d  = GaborConv2d(channel_in, n_competitor, ksize,
                                         stride=2, padding=ksize//2,
                                         init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(n_competitor, n_competitor, ksize,
                                         stride=2, padding=ksize//2,
                                         init_ratio=init_ratio)
        self.argmax   = nn.Softmax(dim=1)   # channel competition
        self.argmax_x = nn.Softmax(dim=2)   # spatial-x competition
        self.argmax_y = nn.Softmax(dim=3)   # spatial-y competition
        self.conv1_1  = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.conv2_1  = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.maxpool  = nn.MaxPool2d(2, 2)
        self.se1      = SELayer(n_competitor)
        self.se2      = SELayer(n_competitor)
        self.weight_chan = weight
        self.weight_spa  = (1 - weight) / 2

    def forward(self, x):
        # 1st order
        x = self.gabor_conv2d(x)
        x_1 = self.weight_chan * self.argmax(x) + self.weight_spa * (
              self.argmax_x(x) + self.argmax_y(x))
        x_1 = self.se1(x_1)
        x_1 = self.maxpool(self.conv1_1(x_1))

        # 2nd order
        x = self.gabor_conv2d2(x)
        x_2 = self.weight_chan * self.argmax(x) + self.weight_spa * (
              self.argmax_x(x) + self.argmax_y(x))
        x_2 = self.se2(x_2)
        x_2 = self.maxpool(self.conv2_1(x_2))

        return torch.cat((x_1.view(x_1.shape[0], -1),
                          x_2.view(x_2.shape[0], -1)), dim=1)


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

    def forward(self, inp, label=None):
        cosine = F.linear(F.normalize(inp), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine) if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine))
        return self.s * cosine


class ccnet(nn.Module):
    """
    CCNet = CB1 // CB2 // CB3 + FC(13152→4096→2048) + Dropout + ArcFace.
    forward() returns (logits, normalised_6144d_contrastive_features).
    get_embedding() returns L2-normalised 2048-d matching embedding.
    """
    def __init__(self, num_classes, weight=0.8, dropout=0.5,
                 arcface_s=30.0, arcface_m=0.50):
        super().__init__()
        self.num_classes = num_classes
        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(
            1,  9, 35, stride=3, padding=17, init_ratio=1,    weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(
            1, 36, 17, stride=3, padding=8,  init_ratio=0.5,  weight=weight, o2=24)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(
            1,  9,  7, stride=3, padding=3,  init_ratio=0.25, weight=weight)
        self.fc       = nn.Linear(13152, 4096)
        self.fc1      = nn.Linear(4096, 2048)
        self.drop     = nn.Dropout(p=dropout)
        self.arclayer = ArcMarginProduct(2048, num_classes,
                                         s=arcface_s, m=arcface_m)

    def forward(self, x, y=None):
        x1 = self.cb1(x); x2 = self.cb2(x); x3 = self.cb3(x)
        x  = torch.cat((x1, x2, x3), dim=1)
        x1 = self.fc(x); x = self.fc1(x1)
        fe = F.normalize(torch.cat((x1, x), dim=1), dim=-1)  # 6144-d contrastive
        x  = self.arclayer(self.drop(x), y)
        return x, fe

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 2048-d embedding for matching."""
        x1 = self.cb1(x); x2 = self.cb2(x); x3 = self.cb3(x)
        x  = torch.cat((x1, x2, x3), dim=1)
        return F.normalize(self.fc1(self.fc(x)), p=2, dim=1)


# ══════════════════════════════════════════════════════════════
#  NORMALISATION  (exact copy from official CCNet)
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


def parse_x_palm_data(data_root, use_scanner=False):
    """
    Load X-Palm dataset (all 206 hand identities, 103 participants × 2 hands).
    Perspective images are always loaded; scanner images (6 spectral bands:
    green, ir, yellow, red, blue, white) are included when use_scanner=True.
    """
    IMG_EXTS = {".jpg",".jpeg",".bmp",".png"}; id2paths = defaultdict(list)
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
    print(f"  [X-Palm/{mode}]")
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
    elif name == "xpalm":
        return lambda: parse_x_palm_data(cfg["x_palm_data_root"],
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
    """Filename includes model class name to avoid cross-model conflicts.
    Cache is stored in base_results_dir."""
    cache_dir    = os.path.abspath(cfg.get("base_results_dir", "./rst_ccnet_all"))
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

class PairedDataset(Dataset):
    """
    Paired-image training dataset for CCNet SupConLoss.
    Returns [img1, img2] where img2 is a different sample of the same identity.
    Matches the official CCNet paired-sampling logic.
    """
    def __init__(self, samples, img_side=128, augment_factor=1):
        self.samples        = samples
        self.augment_factor = augment_factor
        self.label2idxs     = defaultdict(list)
        for i, (_, lab) in enumerate(samples):
            self.label2idxs[lab].append(i)
        self.aug_transform = T.Compose([
            T.Resize(img_side),
            T.RandomChoice([
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.0,1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1),
                T.RandomChoice([
                    T.RandomRotation(10, expand=False,
                                     center=(0.5*img_side, 0.0)),
                    T.RandomRotation(10, expand=False,
                                     center=(0.0, 0.5*img_side)),
                ]),
            ]),
            T.ToTensor(), NormSingleROI(outchannels=1),
        ])

    def __len__(self): return len(self.samples) * self.augment_factor

    def __getitem__(self, index):
        real_idx     = index % len(self.samples)
        path1, label = self.samples[real_idx]
        idxs = self.label2idxs[label]
        idx2 = real_idx
        while idx2 == real_idx and len(idxs) > 1:
            idx2 = random.choice(idxs)
        path2 = self.samples[idx2][0]
        img1  = self.aug_transform(Image.open(path1).convert("L"))
        img2  = self.aug_transform(Image.open(path2).convert("L"))
        return [img1, img2], label


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
#  TRAINING  (CCNet: paired images, CE + SupCon — unchanged)
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, con_criterion,
                  optimizer, device, phase, ce_weight=0.8, con_weight=0.2):
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for datas, target in loader:
            data1 = datas[0].to(device); data2 = datas[1].to(device)
            target = target.to(device)
            if is_train: optimizer.zero_grad()
            output,  fe1 = model(data1, target if is_train else None)
            output2, fe2 = model(data2, target if is_train else None)
            fe      = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
            ce_loss  = criterion(output, target)
            con_loss = con_criterion(fe, target)
            loss     = ce_weight * ce_loss + con_weight * con_loss
            if is_train: loss.backward(); optimizer.step()
            running_loss    += loss.item() * data1.size(0)
            running_correct += output.data.max(1)[1].eq(target).sum().item()
            total           += data1.size(0)
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
    """EER from two score arrays (dot-product similarity, higher=more similar)."""
    if genuine.mean() < impostor.mean():
        genuine = -genuine; impostor = -impostor
    y   = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    s   = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)


def compute_eer(scores_array, n_trials=10, seed=42):
    """
    Returns (eer_all, eer_bal):
      eer_all : EER with ALL impostor pairs (unbalanced reference)
      eer_bal : mean EER over n_trials of 1:1 balanced genuine/impostor sampling
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
    """Returns (eer_all, eer_bal, rank1)."""
    probe_feats,   probe_labels   = extract_features(model, probe_loader,   device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe    = len(probe_feats)
    sim_matrix = probe_feats @ gallery_feats.T

    scores_list, labels_list = [], []
    for i in range(n_probe):
        for j in range(sim_matrix.shape[1]):
            scores_list.append(float(sim_matrix[i, j]))
            labels_list.append(1 if probe_labels[i] == gallery_labels[j] else -1)

    scores_arr       = np.column_stack([scores_list, labels_list])
    eer_all, eer_bal = compute_eer(scores_arr)

    nn_idx  = np.argmax(sim_matrix, axis=1)
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]] for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")

    print(f"  [{tag}]  EER_all={eer_all*100:.4f}%  "
          f"EER_bal={eer_bal*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer_all, eer_bal, rank1


# ══════════════════════════════════════════════════════════════
#  SINGLE EXPERIMENT
# ══════════════════════════════════════════════════════════════

def run_experiment(train_data, test_data, cfg, device=None):
    """
    Train CCNet on train_data, evaluate on test_data using the
    last-epoch checkpoint. Returns (final_eer_bal, final_rank1).
    """
    seed            = cfg["random_seed"]
    results_dir     = cfg["results_dir"]
    img_side        = cfg["img_side"]
    batch_size      = cfg["batch_size"]
    num_epochs      = cfg["num_epochs"]
    lr              = cfg["lr"]
    lr_step         = cfg["lr_step"]
    lr_gamma        = cfg["lr_gamma"]
    comp_weight     = cfg["comp_weight"]
    dropout         = cfg["dropout"]
    arcface_s       = cfg["arcface_s"]
    arcface_m       = cfg["arcface_m"]
    ce_weight       = cfg["ce_weight"]
    con_weight      = cfg["con_weight"]
    temperature     = cfg["temperature"]
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

    # ── data ─────────────────────────────────────────────────────────────
    if same_dataset:
        print(f"  Parsing {train_data} (shared train+test) …")
        all_id2paths = get_parser(train_data, cfg)()
        (train_samples, gallery_samples, probe_samples,
         train_label_map, _) = split_same_dataset(
            all_id2paths, train_sub_ratio, test_gal_ratio, seed)
        num_classes  = len(train_label_map)

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

    train_loader = DataLoader(
        PairedDataset(train_samples, img_side, augment_factor),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Gallery={len(gallery_samples)}  Probe={len(probe_samples)}  Classes={num_classes}")

    # ── model ─────────────────────────────────────────────────────────────
    net = ccnet(num_classes=num_classes, weight=comp_weight,
                dropout=dropout, arcface_s=arcface_s, arcface_m=arcface_m)
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    net = get_or_create_init_weights(net, cfg, num_classes, device)

    criterion     = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=temperature, base_temperature=temperature)
    optimizer     = optim.Adam(net.parameters(), lr=lr)
    scheduler     = lr_scheduler.StepLR(optimizer, lr_step, lr_gamma)

    # ── pre-training evaluation ───────────────────────────────────────────
    _net = net.module if isinstance(net, DataParallel) else net
    pre_eer_all, pre_eer_bal, pre_r1 = evaluate(
        _net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag=f"ep-001_pretrain_{eval_tag_base}")
    last_eer_all = pre_eer_all; last_eer_bal = pre_eer_bal; last_rank1 = pre_r1

    train_losses, train_accs = [], []

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            net, train_loader, criterion, con_criterion,
            optimizer, device, "training",
            ce_weight=ce_weight, con_weight=con_weight)
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

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ts = time.strftime("%H:%M:%S")
            if math.isnan(last_eer_all):
                eer_str = "N/A"
            else:
                eer_str = (f"EER_all={last_eer_all*100:.4f}% | "
                           f"EER_bal={last_eer_bal*100:.4f}%")
            rank1_str = f"{last_rank1:.2f}%" if not math.isnan(last_rank1) else "N/A"
            print(f"  [{ts}] ep {epoch:04d} | loss={t_loss:.4f} | acc={t_acc:.2f}% | "
                  f"{eer_str} | Rank-1={rank1_str}")

        # Save checkpoint at regular intervals and always at the final epoch
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))

    # ── final evaluation (last-epoch checkpoint) ──────────────────────────
    last_path = os.path.join(results_dir, "net_params.pth")
    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(last_path, map_location=device))
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

def print_and_save_table(results, train_datasets, test_datasets, out_path):
    col_w    = 14
    td_label = [t.replace("-","") for t in test_datasets] + ["Avg"]
    header   = f"{'Train\\Test':<14}" + "".join(f"{t:>{col_w}}" for t in td_label)
    sep      = "─" * len(header)
    lines    = []

    for metric_label, idx in [("EER_bal (%)", 0), ("Rank-1 (%)", 1)]:
        lines.append(f"\n{metric_label} Results")
        lines.append(sep); lines.append(header); lines.append(sep)
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


# ══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def main():
    seed = BASE_CONFIG["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir = BASE_CONFIG.get("base_results_dir", "./rst_ccnet_all")
    os.makedirs(base_results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CCNet — Full Cross-Dataset Experiment")
    print(f"  Device       : {device}")
    print(f"  Train sets   : {TRAIN_DATASETS}")
    print(f"  Test  sets   : {TEST_DATASETS}")
    print(f"  Epochs       : {BASE_CONFIG['num_epochs']}")
    print(f"  comp_weight  : {BASE_CONFIG['comp_weight']}  "
          f"(channel vs spatial competition)")
    print(f"  Loss         : {BASE_CONFIG['ce_weight']}×CE + "
          f"{BASE_CONFIG['con_weight']}×SupCon(τ={BASE_CONFIG['temperature']})")
    print(f"  Eval         : last-epoch checkpoint (net_params.pth)")
    print(f"  EER_bal = balanced 1:1 impostor sampling")
    print(f"  EER_all = all impostor pairs (reference)")
    print(f"  Results dir  : {base_results_dir}")
    print(f"{'='*60}\n")

    # ── Loop over all train × test combinations ───────────────────────────
    n_total  = len(TRAIN_DATASETS) * len(TEST_DATASETS)
    n_done   = 0
    results  = {}   # (train, test) → (eer_bal_pct, rank1_pct)
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
                eer_bal, rank1 = run_experiment(
                    train_data, test_data, cfg, device=device)

                results[(train_data, test_data)] = (eer_bal * 100, rank1)
                elapsed = time.time() - t_start
                print(f"\n  ✓  {exp_label}")
                print(f"     EER_bal={eer_bal*100:.4f}%  Rank-1={rank1:.2f}%  "
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

    json_results = {f"{tr}→{te}": list(v) if v else None
                    for (tr, te), v in results.items()}
    with open(os.path.join(base_results_dir, "results_raw.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nRaw results: {os.path.join(base_results_dir, 'results_raw.json')}")


if __name__ == "__main__":
    main()
