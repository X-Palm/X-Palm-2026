"""
CO3Net — Cross-Domain Open-Set Evaluations on X-Palm
==========================================================
All evaluations follow a closed-set protocol: the same subject IDs
appear in both train and test splits.

Settings
────────
  S_scanner         │ Train : roi_perspective (all conditions)
                    │ Gallery: roi_scanner  |  Probe: roi_perspective

  S_scanner_to_persp│ Train : roi_scanner
                    │ Gallery: roi_scanner  |  Probe: roi_perspective

  S_C (×11)         │ Train : roi_perspective (all except C) + roi_scanner
                    │ Gallery: full train set  |  Probe: condition C

Conditions: bf | close | far | fl | jf | pitch | roll | rnd | sf | text | wet
  ("rnd" groups rnd_1 … rnd_5 together)

Scanner spectra kept: green | ir | yellow | pink | white

Model / loss: exact CO3Net (CoordAtt Gabor + CompetitiveBlock,
  ce_weight×CE + con_weight×SupConLoss, paired-image training)
Matching metric: cosine similarity on normalised 2048-d embeddings
Evaluation: EER (single) + Rank-1

Results saved to:
  {BASE_RESULTS_DIR}/setting_scanner/
  {BASE_RESULTS_DIR}/setting_{C}/
  {BASE_RESULTS_DIR}/results_summary.txt
"""

# ==============================================================
#  CONFIG
# ==============================================================
CONFIG = {
    "x_palm_data_root"  : "path to x-palm data",
    "scanner_spectra"      : {"green", "ir", "yellow", "pink", "white"},

    # Open-set split
    "train_id_ratio"       : 0.80,
    "test_gallery_ratio"   : 0.50,

    # Model (official CO3Net)
    "img_side"             : 128,
    "dropout"              : 0.5,
    "arcface_s"            : 20.0,
    "arcface_m"            : 0.30,

    # Loss (official CO3Net)
    "ce_weight"            : 0.8,
    "con_weight"           : 0.2,
    "temperature"          : 0.07,

    # Training
    "batch_size"           : 256,
    "num_epochs"           : 200,
    "lr"                   : 0.001,
    "lr_step"              : 30,
    "lr_gamma"             : 0.6,
    "augment_factor"       : 2,

    # Misc
    "base_results_dir"     : "./rst_co3net_crossdomain_openset",
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

# Shared splits file — all models load this to ensure identical train/test IDs
SPLITS_FILE = "./x_palm_openset_splits.json"
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
#  SUPCONLOSS  (exact official CO3Net — unchanged)
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
#  CO3NET ARCHITECTURE  (exact official CO3Net — unchanged)
# ══════════════════════════════════════════════════════════════

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size,
                 stride=1, padding=0, init_ratio=1):
        super().__init__()
        self.channel_in  = channel_in; self.channel_out = channel_out
        self.kernel_size = kernel_size; self.stride = stride
        self.padding     = padding
        self.init_ratio  = init_ratio if init_ratio > 0 else 1.0
        self.kernel      = 0
        self._SIGMA = 9.2 * self.init_ratio; self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]))
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]))
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False)
        self.f   = nn.Parameter(torch.FloatTensor([self._FREQ]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

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
            -0.5*((gamma*xt)**2+yt**2)/(8*sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2*math.pi*f.view(-1,1,1,1)*xt + psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in,
                                   self.channel_out, self.sigma, self.gamma,
                                   self.theta, self.f, self.psi)
        self.kernel = kernel
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True): super().__init__(); self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x): return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True): super().__init__(); self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x): return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=1):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1  = nn.Conv2d(inp, mip, 1, 1, 0)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, 0)

    def forward(self, x):
        identity = x; n, c, h, w = x.size()
        x_h = self.pool_h(x); x_w = self.pool_w(x).permute(0,1,3,2)
        y   = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0,1,3,2)
        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


class CompetitiveBlock(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()
        self.gabor_conv2d  = GaborConv2d(channel_in, n_competitor, ksize, stride, padding, init_ratio)
        self.gabor_conv2d2 = GaborConv2d(n_competitor, n_competitor, ksize, 1, ksize//2, init_ratio)
        self.cooratt1 = CoordAtt(n_competitor, n_competitor)
        self.cooratt2 = CoordAtt(n_competitor, n_competitor)
        self.a        = nn.Parameter(torch.FloatTensor([1]))
        self.b        = nn.Parameter(torch.FloatTensor([0]))
        self.argmax   = nn.Softmax(dim=1)
        self.conv1    = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool  = nn.MaxPool2d(2, 2)
        self.conv2    = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.cooratt1(self.gabor_conv2d(x))
        x = self.cooratt2(self.gabor_conv2d2(x))
        x = self.argmax((x - self.b) * self.a)
        return self.conv2(self.maxpool(self.conv1(x)))


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight      = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
            phi  = cosine * self.cos_m - sine * self.sin_m
            phi  = (torch.where(cosine > 0, phi, cosine) if self.easy_margin
                    else torch.where(cosine > self.th, phi, cosine - self.mm))
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            return self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine))
        return self.s * F.linear(F.normalize(input), F.normalize(self.weight))


class co3net(nn.Module):
    """CO3Net = CB1//CB2//CB3 + FC(17328→4096→2048) + Dropout + ArcFace."""
    def __init__(self, num_classes, dropout=0.5, arcface_s=20.0, arcface_m=0.30):
        super().__init__()
        self.num_classes = num_classes
        self.cb1  = CompetitiveBlock(1, 9,  35, 3, 17, init_ratio=1,    o2=12)
        self.cb2  = CompetitiveBlock(1, 36, 17, 3, 8,  init_ratio=0.5,  o2=24)
        self.cb3  = CompetitiveBlock(1, 9,  7,  3, 3,  init_ratio=0.25, o2=12)
        self.fc   = nn.Linear(17328, 4096)
        self.fc1  = nn.Linear(4096, 2048)
        self.drop = nn.Dropout(p=dropout)
        self.arclayer = ArcMarginProduct(2048, num_classes, s=arcface_s, m=arcface_m)

    def forward(self, x, y=None):
        x1 = self.cb1(x).view(x.shape[0], -1)
        x2 = self.cb2(x).view(x.shape[0], -1)
        x3 = self.cb3(x).view(x.shape[0], -1)
        x  = torch.cat((x1, x2, x3), dim=1)
        x1 = self.fc(x); x = self.fc1(x1)
        fe = F.normalize(torch.cat((x1, x), dim=1), dim=-1)
        x  = self.arclayer(self.drop(x), y)
        return x, fe

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 2048-d embedding for cosine similarity matching."""
        x1 = self.cb1(x).view(x.shape[0], -1)
        x2 = self.cb2(x).view(x.shape[0], -1)
        x3 = self.cb3(x).view(x.shape[0], -1)
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


def _split_ids(ids, train_ratio, seed):
    """Split a sorted list of IDs into (train_ids, test_ids) with no overlap."""
    rng = random.Random(seed)
    ids = list(ids); rng.shuffle(ids)
    n_train = max(1, int(len(ids) * train_ratio))
    return sorted(ids[:n_train]), sorted(ids[n_train:])


def _all_samples(id2paths, label_map):
    """Flatten id2paths → flat (path, label) list (used for training)."""
    return [(p, label_map[ident])
            for ident, paths in id2paths.items()
            for p in paths]


def _gallery_probe_split(id2paths, label_map, gallery_ratio, rng):
    """50/50 sample-level gallery/probe split — every test ID appears in both."""
    gallery, probe = [], []
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        n_gal = min(n_gal, len(paths) - 1)   # guarantee at least 1 probe
        if len(paths) == 1:                   # edge case: duplicate single image
            gallery.append((paths[0], label_map[ident]))
            probe.append((paths[0], label_map[ident]))
        else:
            for p in paths[:n_gal]: gallery.append((p, label_map[ident]))
            for p in paths[n_gal:]: probe.append((p, label_map[ident]))
    return gallery, probe


# ══════════════════════════════════════════════════════════════
#  PARSERS — OPEN-SET SETTINGS
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
#  TRAIN/TEST ID SPLIT PERSISTENCE
# ══════════════════════════════════════════════════════════════

def generate_all_splits(cond_paths, scanner_paths, train_id_ratio, seed):
    """
    Determine train/test ID splits for all 12 settings and return as a dict.
    This is called once and saved to SPLITS_FILE so every model uses
    identical splits for a fair comparison.
    """
    import random as _random

    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)

    all_persp_ids = sorted(persp_all.keys())
    scanner_ids   = sorted(scanner_paths.keys())
    n_test = len(all_persp_ids) - int(len(all_persp_ids) * train_id_ratio)

    splits = {}

    # S_scanner: test from scanner IDs, train = rest of all perspective IDs
    test_ids  = sorted(_random.Random(seed).sample(scanner_ids, n_test))
    train_ids = sorted(set(all_persp_ids) - set(test_ids))
    splits["S_scanner"] = {"train_ids": train_ids, "test_ids": test_ids}

    # S_scanner_to_persp: train = all scanner IDs, test = no-scanner perspective IDs
    no_scanner_ids = sorted(set(all_persp_ids) - set(scanner_ids))
    splits["S_scanner_to_persp"] = {
        "train_ids": scanner_ids,
        "test_ids" : no_scanner_ids,
    }

    # Paired-condition settings
    for cond_a, cond_b in PAIRED_CONDITIONS:
        paths_a = cond_paths.get(cond_a, {})
        paths_b = cond_paths.get(cond_b, {})
        eligible_ids = sorted(set(paths_a.keys()) & set(paths_b.keys()))
        if not eligible_ids:
            print(f"  [WARN] No IDs with both '{cond_a}' and '{cond_b}' — skipping split")
            continue
        # n_test fixed at 20% of ALL perspective IDs, capped by eligible count
        n_test    = len(all_persp_ids) - int(len(all_persp_ids) * train_id_ratio)
        n_test    = min(n_test, len(eligible_ids))
        test_ids  = sorted(_random.Random(seed).sample(eligible_ids, n_test))
        train_ids = sorted(set(all_persp_ids) - set(test_ids))  # 190 - n_test
        key = f"S_{cond_a}_{cond_b}"
        splits[key] = {"train_ids": train_ids, "test_ids": test_ids}

    return splits


def load_or_generate_splits(cond_paths, scanner_paths, train_id_ratio, seed):
    """
    Load splits from SPLITS_FILE if it exists; otherwise generate, save, and return them.
    Prints a clear message so the user knows which path was taken.
    """
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE) as f:
            splits = json.load(f)
        print(f"  Loaded existing ID splits from: {SPLITS_FILE}")
        # Print summary
        for key, val in splits.items():
            print(f"    {key:<30}  train={len(val['train_ids'])}  test={len(val['test_ids'])}")
    else:
        print(f"  Generating ID splits (seed={seed}) → {SPLITS_FILE}")
        splits = generate_all_splits(cond_paths, scanner_paths, train_id_ratio, seed)
        with open(SPLITS_FILE, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"  Splits saved to: {SPLITS_FILE}")
        for key, val in splits.items():
            print(f"    {key:<30}  train={len(val['train_ids'])}  test={len(val['test_ids'])}")
    return splits

def parse_setting_scanner(cond_paths, scanner_paths,
                          splits, gallery_ratio, seed):
    """
    S_scanner — uses pre-computed splits for fair cross-model comparison.
    Train (80 %): ALL perspective images for train IDs
    Test  (20 %): scanner images for test IDs → 50/50 gallery/probe
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
        test_label_map, gallery_ratio, rng)

    _print_stats("S_scanner | Perspective (train IDs) -> Scanner (test IDs)",
                 len(train_ids), len(test_ids), len(train_samples),
                 len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_scanner_to_perspective(cond_paths, scanner_paths,
                                         splits, gallery_ratio, seed):
    """
    S_scanner_to_persp — uses pre-computed splits.
    Train: ALL scanner images for scanner IDs
    Test : ALL perspective images for no-scanner IDs → 50/50 gallery/probe
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
        test_label_map, gallery_ratio, rng)

    _print_stats("S_scanner_to_persp | Scanner (all) -> Perspective (no-scanner IDs)",
                 len(train_ids), len(test_ids), len(train_samples),
                 len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_paired_conditions(cond_a, cond_b, cond_paths, scanner_paths,
                                    splits, seed):
    """
    S_{A}_{B} — uses pre-computed splits for fair cross-model comparison.
    Train: perspective (all except cond_A and cond_B) + scanner, train IDs only
    Test : 1 image from cond_A -> gallery, 1 image from cond_B -> probe, test IDs only
    """
    rng = random.Random(seed)

    paths_a = cond_paths.get(cond_a, {})
    paths_b = cond_paths.get(cond_b, {})

    train_ids = splits["train_ids"]
    test_ids  = splits["test_ids"]

    train_label_map = {ident: i for i, ident in enumerate(train_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(test_ids)}
    num_train_cls   = len(train_ids)

    # Train: all perspective except both test conditions + scanner, train IDs
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

    # Test: 1 sample from cond_A, 1 from cond_B, randomly assigned to gallery/probe
    gallery_samples = []
    probe_samples   = []
    for ident in test_ids:
        label  = test_label_map[ident]
        a_imgs = list(paths_a.get(ident, [])); rng.shuffle(a_imgs)
        b_imgs = list(paths_b.get(ident, [])); rng.shuffle(b_imgs)
        if not a_imgs or not b_imgs:
            continue
        # Random assignment: coin flip decides which condition goes to gallery
        if rng.random() < 0.5:
            gallery_samples.append((a_imgs[0], label))
            probe_samples.append((b_imgs[0], label))
        else:
            gallery_samples.append((b_imgs[0], label))
            probe_samples.append((a_imgs[0], label))

    _print_stats(
        f"S_{cond_a}_{cond_b} | Perspective (not {cond_a}/{cond_b}) + Scanner"
        f" -> {cond_a}/{cond_b}",
        len(train_ids), len(test_ids), len(train_samples),
        len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


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
    CO3Net training dataset: returns two augmented views of the same identity
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
#  TRAINING  (paired images, CE + SupCon — unchanged from official)
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, con_criterion,
                  optimizer, device, phase, ce_weight=0.8, con_weight=0.2):
    """CO3Net composite loss: ce_weight×CE + con_weight×SupConLoss"""
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
#  EVALUATION  (cosine similarity, single EER, argmax Rank-1)
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
    """Cosine-similarity evaluation. Returns (eer, rank1)."""
    probe_feats,   probe_labels   = extract_features(model, probe_loader,   device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    n_probe    = len(probe_feats)
    sim_matrix = probe_feats @ gallery_feats.T   # cosine sim (embeddings L2-normalised)

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
#  EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════

def run_experiment(train_samples, gallery_samples, probe_samples,
                   num_classes, cfg, results_dir, device):
    """Train CO3Net and evaluate. Returns (final_eer, final_rank1)."""
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
    net = co3net(num_classes=num_classes, dropout=dropout,
                 arcface_s=arcface_s, arcface_m=arcface_m)
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
    pre_eer, pre_r1 = evaluate(_net, probe_loader, gallery_loader,
                                device, out_dir=rst_eval, tag="ep-001_pretrain")
    best_eer = pre_eer; last_eer = pre_eer; last_rank1 = pre_r1
    torch.save(_net.state_dict(),
               os.path.join(results_dir, "net_params_best_eer.pth"))

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
            cur_eer, cur_rank1 = evaluate(
                _net, probe_loader, gallery_loader,
                device, out_dir=rst_eval, tag=f"ep{epoch:04d}")
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
            print(f"  [{ts}] ep {epoch:04d} | loss={t_loss:.4f} | acc={t_acc:.2f}% | "
                  f"EER={eer_str}  Rank-1={rank1_str}")

        if epoch % save_every == 0 or epoch == num_epochs - 1:
            torch.save(_net.state_dict(),
                       os.path.join(results_dir, "net_params.pth"))

    # ── Final evaluation (best checkpoint) ────────────────────────────────
    best_path = os.path.join(results_dir, "net_params_best_eer.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(results_dir, "net_params.pth")
    eval_net = net.module if isinstance(net, DataParallel) else net
    eval_net.load_state_dict(torch.load(best_path, map_location=device))
    final_eer, final_rank1 = evaluate(
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
    lines = ["\nCross-Domain Open-Set Results — Palm-Auth (CompNet)", sep, header, sep]

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
    cfg  = CONFIG
    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir = cfg["base_results_dir"]
    os.makedirs(base_results_dir, exist_ok=True)

    train_ratio  = cfg["train_id_ratio"]
    gallery_ratio = cfg["test_gallery_ratio"]

    print(f"\n{'='*60}")
    print(f"  CompNet — Cross-Domain Open-Set (Palm-Auth)")
    print(f"  Protocol  : open set (80/20 ID split, no overlap)")
    print(f"  Train IDs : {train_ratio*100:.0f}%   Test IDs: {(1-train_ratio)*100:.0f}%")
    print(f"  Gallery/Probe: {gallery_ratio*100:.0f}/{(1-gallery_ratio)*100:.0f} sample split")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {cfg['num_epochs']}")
    print(f"  Settings  : 2 scanner + {len(PAIRED_CONDITIONS)} paired-condition")
    print(f"  Results   : {base_results_dir}")
    print(f"{'='*60}")

    print("\n  Scanning dataset …")
    cond_paths    = _collect_perspective(cfg["palm_auth_data_root"])
    scanner_paths = _collect_scanner(cfg["palm_auth_data_root"],
                                     cfg["scanner_spectra"])
    print(f"  Perspective conditions found : {sorted(cond_paths.keys())}")
    print(f"  Scanner identities found     : {len(scanner_paths)}")

    # ── Load or generate shared ID splits ───────────────────────────────────
    all_splits = load_or_generate_splits(
        cond_paths, scanner_paths, train_ratio, seed)

    # ── Build settings list ───────────────────────────────────────────────────
    SETTINGS = []

    SETTINGS.append({
        "tag"        : "setting_scanner",
        "label"      : "S_scanner",
        "train_desc" : "Perspective (all, train IDs)",
        "test_desc"  : "Scanner (test IDs)",
        "parser"     : lambda: parse_setting_scanner(
                           cond_paths, scanner_paths,
                           all_splits["S_scanner"], gallery_ratio, seed),
    })

    SETTINGS.append({
        "tag"        : "setting_scanner_to_persp",
        "label"      : "S_scanner_to_persp",
        "train_desc" : "Scanner (all scanner IDs)",
        "test_desc"  : "Perspective (no-scanner IDs)",
        "parser"     : lambda: parse_setting_scanner_to_perspective(
                           cond_paths, scanner_paths,
                           all_splits["S_scanner_to_persp"], gallery_ratio, seed),
    })

    conditions_found = sorted(cond_paths.keys())
    for cond_a, cond_b in PAIRED_CONDITIONS:
        if cond_a not in conditions_found or cond_b not in conditions_found:
            print(f"  [WARN] '{cond_a}' or '{cond_b}' not found in data — skipping")
            continue
        ca, cb = cond_a, cond_b
        split_key = f"S_{ca}_{cb}"
        if split_key not in all_splits:
            print(f"  [WARN] No split found for {split_key} — skipping")
            continue
        SETTINGS.append({
            "tag"        : f"setting_{ca}_{cb}",
            "label"      : f"S_{ca}_{cb}",
            "train_desc" : f"Perspective (not {ca}/{cb}) + Scanner (train IDs)",
            "test_desc"  : f"{ca}/{cb} (test IDs)",
            "parser"     : (lambda ca=ca, cb=cb: parse_setting_paired_conditions(
                                ca, cb, cond_paths, scanner_paths,
                                all_splits[f"S_{ca}_{cb}"], seed)),
        })

        print(f"\n  Total settings to run : {len(SETTINGS)}")

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
            eer, rank1 = run_experiment(
                train_s, gal_s, probe_s, n_cls, cfg, results_dir, device)
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
                                 "test_desc": s["test_desc"], "eer": None, "rank1": None})

    print(f"\n\n{'='*60}")
    print(f"  ALL {len(SETTINGS)} SETTINGS COMPLETE")
    print(f"{'='*60}")
    print_and_save_summary(
        all_results,
        os.path.join(base_results_dir, "results_summary.txt"))


if __name__ == "__main__":
    main()
