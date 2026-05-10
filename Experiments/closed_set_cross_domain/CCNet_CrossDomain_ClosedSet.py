"""
CCNet — Cross-Domain Closed-Set Evaluations on X-Palm
========================================================
All evaluations follow a closed-set protocol: the same subject IDs
appear in both train and test splits. Gallery and probe are both drawn
from the TEST domain (not the training domain), split at the sample level.

Settings (12 total)
────────────────────
  S_scanner         │ Train : roi_perspective (all conditions, 206 IDs)
                    │ Gallery: 50% of scanner samples  (shared IDs)
                    │ Probe  : 50% of scanner samples  (shared IDs)

  S_scanner_to_persp│ Train : roi_scanner (shared IDs)
                    │ Gallery: 50% of perspective samples (shared IDs)
                    │ Probe  : 50% of perspective samples (shared IDs)

  S_(A,B) (×10)     │ Train : roi_perspective (all except A and B) + roi_scanner
                    │ Gallery: ALL condition A images  (first test domain)
                    │ Probe  : ALL condition B images  (second test domain)

Paired conditions:
  (wet,text) (wet,rnd) (rnd,text) (sf,roll) (jf,pitch)
  (bf,far) (roll,close) (far,jf) (fl,sf) (roll,pitch)

Scanner spectra kept: green | ir | yellow | red | blue | white

Model / loss: CCNet — CE + SupConLoss
Matching metric: cosine similarity on normalised 2048-d embeddings
EER: EER_all + EER_bal

Gallery/probe splits for S_scanner and S_scanner_to_persp are saved to
  x_palm_closedset_splits.json on first run and reused by all models.

Results saved to:
  {BASE_RESULTS_DIR}/setting_scanner/
  {BASE_RESULTS_DIR}/setting_{A}_{B}/
  {BASE_RESULTS_DIR}/results_summary.txt

Note: final evaluation uses the last-epoch checkpoint (net_params.pth).
"""


# ==============================================================
#  CONFIG
# ==============================================================
CONFIG = {
    "x_palm_data_root"     : "path to X-Palm data",
    "scanner_spectra"      : {"green", "ir", "yellow", "red", "blue", "white"},
    "test_gallery_ratio"   : 0.50,

    # Model (official CCNet)
    "img_side"             : 128,
    "comp_weight"          : 0.8,
    "dropout"              : 0.5,
    "arcface_s"            : 20.0,
    "arcface_m"            : 0.30,

    # Loss (official CCNet)
    "ce_weight"            : 0.8,
    "con_weight"           : 0.2,
    "temperature"          : 0.07,

    # Training
    "batch_size"           : 256,
    "num_epochs"           : 200,
    "lr"                   : 0.001,
    "lr_step"              : 17,
    "lr_gamma"             : 0.8,
    "augment_factor"       : 2,

    # Misc
    "base_results_dir"     : "./rst_ccnet_crossdomain",
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

warnings.filterwarnings("ignore")

IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ══════════════════════════════════════════════════════════════
#  SUPCONLOSS  (exact official CCNet/loss.py — unchanged)
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
#  CCNET ARCHITECTURE  (exact official CCNet — unchanged)
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
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 weight, init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor_conv2d  = GaborConv2d(channel_in, n_competitor, ksize,
                                         stride=2, padding=ksize//2,
                                         init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(n_competitor, n_competitor, ksize,
                                         stride=2, padding=ksize//2,
                                         init_ratio=init_ratio)
        self.argmax   = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        self.conv1_1  = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.conv2_1  = nn.Conv2d(n_competitor, o1 // 2, 5, 2, 0)
        self.maxpool  = nn.MaxPool2d(2, 2)
        self.se1      = SELayer(n_competitor)
        self.se2      = SELayer(n_competitor)
        self.weight_chan = weight
        self.weight_spa  = (1 - weight) / 2

    def forward(self, x):
        x = self.gabor_conv2d(x)
        x_1 = self.weight_chan * self.argmax(x) + self.weight_spa * (
              self.argmax_x(x) + self.argmax_y(x))
        x_1 = self.se1(x_1)
        x_1 = self.maxpool(self.conv1_1(x_1))
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
    def __init__(self, num_classes, weight=0.8, dropout=0.5,
                 arcface_s=20.0, arcface_m=0.30):
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
        fe = F.normalize(torch.cat((x1, x), dim=1), dim=-1)
        x  = self.arclayer(self.drop(x), y)
        return x, fe

    @torch.no_grad()
    def get_embedding(self, x):
        x1 = self.cb1(x); x2 = self.cb2(x); x3 = self.cb3(x)
        x  = torch.cat((x1, x2, x3), dim=1)
        return F.normalize(self.fc1(self.fc(x)), p=2, dim=1)


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
    "rnd" covers rnd_1…rnd_5 (parts[2] == "rnd" for all).
    X-Palm contains 206 perspective hand identities (103 participants × 2 hands).
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
    Lowercase side so keys match perspective: "{id}_{side_lowercase}"
    Spectra collected: green, ir, yellow, red, blue, white.
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
    """Flatten id2paths → flat (path, label) list."""
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
    X-Palm has 206 perspective IDs (103 participants × 2 hands).
    """
    rng = random.Random(seed)

    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    all_persp_ids = sorted(persp_all.keys())      # 206 IDs (103 participants × 2 hands)
    scanner_ids   = sorted(scanner_paths.keys())  # shared IDs with scanner data

    splits = {}

    # S_scanner: 50/50 split of scanner samples for the shared scanner IDs
    splits["S_scanner"] = _make_gallery_probe_split(
        {i: scanner_paths[i] for i in scanner_ids},
        gallery_ratio, rng)

    # S_scanner_to_persp: 50/50 split of perspective samples for the shared scanner IDs
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
    Train   : ALL perspective images for ALL 206 perspective IDs
    Gallery : 50% of scanner samples for the shared IDs (pre-computed)
    Probe   : 50% of scanner samples for the shared IDs (pre-computed)
    Closed-set: scanner test IDs are a subset of the 206 training IDs.
    """
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    all_persp_ids = sorted(persp_all.keys())      # 206 IDs (103 participants × 2 hands)
    scanner_ids   = sorted(scanner_paths.keys())  # shared IDs with scanner data

    # Training label map uses ALL 206 perspective IDs
    train_label_map = {ident: i for i, ident in enumerate(all_persp_ids)}
    # Test label map uses only the scanner IDs
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    num_train_cls   = len(all_persp_ids)

    train_samples = [(p, train_label_map[i])
                     for i in all_persp_ids for p in persp_all[i]]

    split = stored_splits["S_scanner"]
    gallery_samples, probe_samples = _gallery_probe_split_from_stored(
        {i: scanner_paths[i] for i in scanner_ids}, test_label_map, split)

    _print_stats(
        f"S_scanner | Perspective/{len(all_persp_ids)} (train) → "
        f"Scanner/{len(scanner_ids)} 50/50 (test)",
        len(all_persp_ids), len(scanner_ids), len(train_samples),
        len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_scanner_to_perspective(cond_paths, scanner_paths,
                                         stored_splits, seed):
    """
    S_scanner_to_persp — Scanner (train) → Perspective (gallery + probe)
    ──────────────────────────────────────────────────────────────────────
    Train   : ALL scanner images for ALL shared scanner IDs
    Gallery : 50% of perspective samples for the shared IDs (pre-computed)
    Probe   : 50% of perspective samples for the shared IDs (pre-computed)
    Closed-set: same shared IDs in train and test.
    """
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    scanner_ids = sorted(scanner_paths.keys())  # shared IDs with scanner data

    train_label_map = {ident: i for i, ident in enumerate(scanner_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    num_train_cls   = len(scanner_ids)

    train_samples = [(p, train_label_map[i])
                     for i in scanner_ids for p in scanner_paths[i]]

    split = stored_splits["S_scanner_to_persp"]
    gallery_samples, probe_samples = _gallery_probe_split_from_stored(
        {i: persp_all[i] for i in scanner_ids}, test_label_map, split)

    _print_stats(
        f"S_scanner_to_persp | Scanner/{len(scanner_ids)} (train) → "
        f"Perspective/{len(scanner_ids)} 50/50 (test)",
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

class PairedDataset(Dataset):
    """
    CCNet training dataset: returns two augmented views of the same identity
    for SupConLoss. Falls back to self-pairing when a class has only 1 sample.
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
#  TRAINING  (CCNet composite loss — unchanged from official)
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, con_criterion,
                  optimizer, device, phase, ce_weight=0.8, con_weight=0.2):
    """
    CCNet composite loss: ce_weight×CE + con_weight×SupConLoss
    Each batch item is a (img1, img2, label) triple from PairedDataset.
    """
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
            fe       = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
            ce_loss  = criterion(output, target)
            con_loss = con_criterion(fe, target)
            loss     = ce_weight * ce_loss + con_weight * con_loss
            if is_train: loss.backward(); optimizer.step()
            running_loss    += loss.item() * data1.size(0)
            running_correct += output.data.max(1)[1].eq(target).sum().item()
            total           += data1.size(0)
    return running_loss / max(total, 1), 100.0 * running_correct / max(total, 1)


# ══════════════════════════════════════════════════════════════
#  EVALUATION  (cosine similarity, EER_all + EER_bal, argmax Rank-1)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval(); feats, labels = [], []
    for imgs, labs in loader:
        feats.append(model.get_embedding(imgs.to(device)).cpu().numpy())
        labels.append(labs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _single_eer(genuine, impostor):
    if genuine.mean() < impostor.mean():
        genuine = -genuine; impostor = -impostor
    y   = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    s   = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)


def compute_eer(scores_array, n_trials=10, seed=42):
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
    """Cosine-similarity evaluation. Returns (eer_all, eer_bal, rank1)."""
    probe_feats,   probe_labels   = extract_features(model, probe_loader,   device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe    = len(probe_feats)
    sim_matrix = probe_feats @ gallery_feats.T   # cosine sim (embeddings normalised)

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
#  EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════

def run_experiment(train_samples, gallery_samples, probe_samples,
                   num_classes, cfg, results_dir, device):
    """
    Train CCNet and evaluate using the last-epoch checkpoint.
    Returns (final_eer_bal, final_rank1).
    """
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
    comp_weight    = cfg["comp_weight"]
    dropout        = cfg["dropout"]
    arcface_s      = cfg["arcface_s"]
    arcface_m      = cfg["arcface_m"]
    ce_weight      = cfg["ce_weight"]
    con_weight     = cfg["con_weight"]
    temperature    = cfg["temperature"]

    train_loader = DataLoader(
        PairedDataset(train_samples, img_side, augment_factor),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    net = ccnet(num_classes=num_classes, weight=comp_weight,
                dropout=dropout, arcface_s=arcface_s, arcface_m=arcface_m)
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    net = get_or_create_init_weights(
        net, num_classes,
        cache_dir = cfg["base_results_dir"],
        device    = device)

    criterion     = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=temperature, base_temperature=temperature)
    optimizer     = optim.Adam(net.parameters(), lr=cfg["lr"])
    scheduler     = lr_scheduler.StepLR(optimizer, cfg["lr_step"], cfg["lr_gamma"])

    # ── Pre-training baseline ─────────────────────────────────────────────
    _net = net.module if isinstance(net, DataParallel) else net
    pre_eer_all, pre_eer_bal, pre_r1 = evaluate(
        _net, probe_loader, gallery_loader,
        device, out_dir=rst_eval, tag="ep-001_pretrain")
    last_eer_all = pre_eer_all
    last_eer_bal = pre_eer_bal
    last_rank1   = pre_r1

    train_losses, train_accs = [], []

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        t_loss, t_acc = run_one_epoch(
            net, train_loader, criterion, con_criterion,
            optimizer, device, "training",
            ce_weight=ce_weight, con_weight=con_weight)
        scheduler.step()
        train_losses.append(t_loss); train_accs.append(t_acc)
        _net = net.module if isinstance(net, DataParallel) else net

        if (epoch % eval_every == 0 and epoch > 0) or epoch == num_epochs - 1:
            cur_eer_all, cur_eer_bal, cur_rank1 = evaluate(
                _net, probe_loader, gallery_loader,
                device, out_dir=rst_eval, tag=f"ep{epoch:04d}")
            last_eer_all = cur_eer_all
            last_eer_bal = cur_eer_bal
            last_rank1   = cur_rank1

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            ts = time.strftime("%H:%M:%S")
            eer_str = (f"EER_all={last_eer_all*100:.4f}% | "
                       f"EER_bal={last_eer_bal*100:.4f}%"
                       if not math.isnan(last_eer_all) else "N/A")
            rank1_str = f"{last_rank1:.2f}%" if not math.isnan(last_rank1) else "N/A"
            print(f"  [{ts}] ep {epoch:04d} | loss={t_loss:.4f} | acc={t_acc:.2f}% | "
                  f"{eer_str} | Rank-1={rank1_str}")

        # Save checkpoint at regular intervals and always at the final epoch
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))

    # ── Final evaluation (last-epoch checkpoint) ───────────────────────────
    last_path = os.path.join(results_dir, "net_params.pth")
    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(last_path, map_location=device))
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
    lines = ["\nCross-Domain Closed-Set Results — X-Palm (CCNet)", sep, header, sep]

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

    device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir = cfg["base_results_dir"]
    os.makedirs(base_results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CCNet — Cross-Domain Closed-Set (X-Palm)")
    print(f"  Protocol : closed set (shared IDs in train & test)")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {cfg['num_epochs']}")
    print(f"  Loss     : {cfg['ce_weight']}×CE + "
          f"{cfg['con_weight']}×SupCon(τ={cfg['temperature']})")
    print(f"  Matching : cosine similarity (normalised 2048-d embeddings)")
    print(f"  Eval     : last-epoch checkpoint (net_params.pth)")
    print(f"  Settings  : 2 scanner + 10 paired-condition")
    print(f"  Results  : {base_results_dir}")
    print(f"{'='*60}")

    # ── Pre-collect data once ─────────────────────────────────────────────
    print("\n  Scanning dataset …")
    cond_paths    = _collect_perspective(cfg["x_palm_data_root"])
    scanner_paths = _collect_scanner(cfg["x_palm_data_root"],
                                     cfg["scanner_spectra"])
    print(f"  Perspective conditions found : {sorted(cond_paths.keys())}")
    print(f"  Scanner identities found     : {len(scanner_paths)}")

    # ── Load or generate shared gallery/probe splits ──────────────────────
    all_splits = load_or_generate_closedset_splits(
        cond_paths, scanner_paths, cfg["test_gallery_ratio"], seed)

    # ── Build settings list ───────────────────────────────────────────────
    SETTINGS = []

    SETTINGS.append({
        "tag"        : "setting_scanner",
        "label"      : "S_scanner",
        "train_desc" : "Perspective (all 206 IDs)",
        "test_desc"  : "Scanner 50/50 gallery/probe",
        "parser"     : lambda: parse_setting_scanner(
                           cond_paths, scanner_paths, all_splits, seed),
    })

    SETTINGS.append({
        "tag"        : "setting_scanner_to_persp",
        "label"      : "S_scanner_to_persp",
        "train_desc" : "Scanner (all scanner IDs)",
        "test_desc"  : "Perspective 50/50 gallery/probe",
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
