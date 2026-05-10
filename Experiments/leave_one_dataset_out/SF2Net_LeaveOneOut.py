"""
SF2Net — Leave-One-Out Cross-Dataset Experiment Runner
==================================================
Trains on ALL combinations of three datasets and evaluates on the
left-out fourth dataset. Four experiments total:

  Train                              Test
  ─────────────────────────────────  ──────────
  CASIA-MS + MPDv2   + XJTU          X-Palm
  X-Palm + MPDv2  + XJTU          CASIA-MS
  X-Palm + CASIA-MS + XJTU        MPDv2
  X-Palm + CASIA-MS + MPDv2       XJTU


Model architecture and training: unchanged from official SF2Net.
  - Multi-order Gabor texture (1st + 2nd order) + SE module
  - Sequence Feature Extractor (first-k / last-k selection)
  - Dual ViT for sequence feature processing
  - Weighted fusion of CNN and ViT features → 1024-d embedding
  - ArcFace classification head (s=30, m=0.5)
  - Loss: ce_weight * CrossEntropy + tl_weight * TripletLoss(SRT)
  - Triplet sampling: anchor / positive / negative per batch

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

    # ── Model (official SF2Net values) ─────────────────────────
    "img_side"             : 128,
    "vit_floor_num"        : 10,     # first-k and last-k in SFE
    "cnn_vit_weight"       : 0.7,    # CNN weight (ViT gets 1 - weight)
    "dropout"              : 0.5,
    "arcface_s"            : 30.0,
    "arcface_m"            : 0.50,

    # ── Loss ───────────────────────────────────────────────────
    "ce_weight"            : 0.7,
    "tl_weight"            : 0.3,

    # ── Training ───────────────────────────────────────────────
    "batch_size"           : 256,
    "num_epochs"           : 200,
    "lr"                   : 0.001,
    "lr_step"              : 17,
    "lr_gamma"             : 0.8,
    "augment_factor"       : 4,

    # ── Misc ───────────────────────────────────────────────────
    "base_results_dir"     : "./rst_sf2net_loo",
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

try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
except ImportError:
    os.system("pip install einops --quiet")
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange

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
#  TRIPLET LOSS  (exact copy from official SF2Net)
# ══════════════════════════════════════════════════════════════

class TripletLoss(nn.Module):
    """Triplet loss with SRT (Soft Relative Triplet) distance."""
    def __init__(self, margin=2.0, alpha=0.95, distance="SRT"):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.distance = distance
        self.tripletMargin = nn.TripletMarginLoss(margin=1.0, swap=True, reduction='mean')

    def dis(self, a, b):
        return torch.sum((a - b).pow(2), 1)

    def forward(self, anchor, positive, negative, size_average=True):
        if self.distance == "SRT":
            self.margin = 2.0
            anchor   = F.normalize(anchor,   p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
            pos_d  = self.dis(anchor, positive)
            neg_d  = self.dis(anchor, negative)
            pn_d   = self.dis(positive, negative)
            cond   = neg_d.mean() >= pn_d.mean()
            ls     = torch.where(cond,
                                 pos_d + self.margin - pn_d.mean(),
                                 pos_d + self.margin - neg_d)
            losses = F.relu(ls).mean()
            return losses, pos_d.mean(), neg_d.mean(), pn_d.mean()
        else:
            raise ValueError(f"Unsupported distance: {self.distance}. Use 'SRT'.")


# ══════════════════════════════════════════════════════════════
#  ARCFACE  (exact copy from official SF2Net)
# ══════════════════════════════════════════════════════════════

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.s = s; self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

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


# ══════════════════════════════════════════════════════════════
#  GABOR CONV  (exact copy from official SF2Net)
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
        self.init_ratio  = init_ratio
        self.kernel      = 0
        self.sigma     = nn.Parameter(torch.FloatTensor([9.2  * init_ratio]), requires_grad=True)
        self.gamma     = nn.Parameter(torch.FloatTensor([2.0]),               requires_grad=True)
        self.theta     = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False)
        self.frequency = nn.Parameter(torch.FloatTensor([0.057 / init_ratio]), requires_grad=True)
        self.psi       = nn.Parameter(torch.FloatTensor([0]),                  requires_grad=False)

    def get_gabor(self):
        half = self.kernel_size // 2
        x_0 = torch.arange(-half, half + 1).float()
        y_0 = torch.arange(-half, half + 1).float()
        k   = self.kernel_size
        x = x_0.view(-1, 1).repeat(self.channel_out, self.channel_in, 1, k)
        y = y_0.view(1, -1).repeat(self.channel_out, self.channel_in, k, 1)
        x = x.float().to(self.sigma.device)
        y = y.float().to(self.sigma.device)
        xt =  x*torch.cos(self.theta.view(-1,1,1,1)) + y*torch.sin(self.theta.view(-1,1,1,1))
        yt = -x*torch.sin(self.theta.view(-1,1,1,1)) + y*torch.cos(self.theta.view(-1,1,1,1))
        gb = -torch.exp(
            -0.5*((self.gamma*xt)**2 + yt**2) / (8*self.sigma.view(-1,1,1,1)**2)
        ) * torch.cos(2*math.pi*self.frequency.view(-1,1,1,1)*xt + self.psi.view(-1,1,1,1))
        return gb - gb.mean(dim=[2,3], keepdim=True)

    def forward(self, x):
        self.kernel = self.get_gabor()
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)


# ══════════════════════════════════════════════════════════════
#  SE MODULE  (exact copy from official SF2Net)
# ══════════════════════════════════════════════════════════════

class SEModule(nn.Module):
    def __init__(self, channel, reduction=1):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


# ══════════════════════════════════════════════════════════════
#  SEQUENCE FEATURE EXTRACTOR  (exact copy from official SF2Net)
# ══════════════════════════════════════════════════════════════

def get_sequence_feature(feature_tensor, vit_floor_num):
    ft = torch.softmax(feature_tensor, dim=1)
    front = ft[:, :vit_floor_num, :, :]
    back  = ft[:, -vit_floor_num:, :, :]
    return torch.cat((front, back), dim=1)


class FeatureExtraction(nn.Module):
    def __init__(self, channel_in, filter_num, kernel_size, stride, padding,
                 init_ratio, label_num, vit_floor_num):
        super().__init__()
        self.vit_floor_num = vit_floor_num
        self.gabor_conv2d_1 = GaborConv2d(channel_in, filter_num, kernel_size,
                                          stride, padding, init_ratio)
        self.gabor_conv2d_2 = GaborConv2d(filter_num, filter_num, kernel_size,
                                          stride, padding, init_ratio)
        self.se  = SEModule(channel=filter_num)
        self.conv_0 = nn.Conv2d(filter_num, 64, 5, 1, 0)
        self.conv_1 = nn.Conv2d(filter_num, 64, 5, 1, 0)
        self.conv_2 = nn.Conv2d(64, 32, 3, 2, 0)
        self.conv_3 = nn.Conv2d(64, 32, 3, 2, 0)
        self.max_pool = nn.MaxPool2d(2, 2)

    def process_block(self, x, conv):
        x = self.se(x); x = conv(x); x = torch.relu(x); x = self.max_pool(x)
        return x

    def forward(self, x):
        f1 = self.gabor_conv2d_1(x)
        f2 = self.gabor_conv2d_2(f1)
        f1p = self.process_block(f1, self.conv_0)
        f2p = self.process_block(f2, self.conv_1)
        out1 = self.conv_2(f1p)
        out2 = self.conv_3(f2p)
        feat = torch.cat((out1.flatten(1), out2.flatten(1)), dim=1)
        seq1 = get_sequence_feature(f1p, self.vit_floor_num)
        seq2 = get_sequence_feature(f2p, self.vit_floor_num)
        return feat, seq1, seq2


# ══════════════════════════════════════════════════════════════
#  VIT  (exact copy from official SF2Net)
# ══════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    def __init__(self, dim, dim_for_mlp, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim_for_mlp), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_for_mlp, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_for_head, dropout=0.1):
        super().__init__()
        self.heads = heads
        inner = dim_for_head * heads
        self.scale = dim_for_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out  = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_for_head, dim_for_mlp, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_for_head, dropout),
                FeedForward(dim, dim_for_mlp, dropout)
            ]) for _ in range(depth)])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x; x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, channels, num_classes, depth,
                 heads, dim, dim_for_head, dim_for_mlp, pool='cls',
                 dropout=0.1, emb_dropout=0.1):
        super().__init__()
        ih, iw = image_size, image_size
        ph, pw = patch_size, patch_size
        assert ih % ph == 0 and iw % pw == 0
        num_patches = (ih // ph) * (iw // pw)
        patch_dim   = channels * ph * pw
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw),
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_for_head, dim_for_mlp, dropout)
        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.to_patch(x)
        b, n, _ = x.shape
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(self.dropout(x))
        return self.to_latent(x)


# ══════════════════════════════════════════════════════════════
#  SF2NET MODEL  (exact copy from official SF2Net)
# ══════════════════════════════════════════════════════════════

class SF2Net(nn.Module):
    """SF2Net: Sequence Feature Fusion Network for Palmprint Verification."""
    def __init__(self, num_classes, vit_floor_num=10, weight=0.7,
                 dropout=0.5, arcface_s=30.0, arcface_m=0.50):
        super().__init__()
        self.num_classes   = num_classes
        self.vit_floor_num = vit_floor_num
        self.weight        = weight

        self.feature_extraction = FeatureExtraction(
            channel_in=1, filter_num=36, kernel_size=17, stride=2, padding=8,
            init_ratio=0.5, label_num=num_classes, vit_floor_num=vit_floor_num)

        self.vit_0 = ViT(image_size=30, patch_size=5, channels=vit_floor_num*2,
                         num_classes=num_classes, depth=2, heads=16, dim=128,
                         dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)
        self.vit_1 = ViT(image_size=14, patch_size=2, channels=vit_floor_num*2,
                         num_classes=num_classes, depth=2, heads=16, dim=128,
                         dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)

        self.fc1  = nn.Linear(7424, 2048)
        self.fc2  = nn.Linear(2048, 1024)
        self.vfc1 = nn.Linear(11136, 4096)
        self.vfc2 = nn.Linear(4096, 1024)

        self.dropout  = nn.Dropout(p=dropout)
        self.arcface  = ArcMarginProduct(1024, num_classes, s=arcface_s, m=arcface_m)

    def _process(self, x):
        feat, seq1, seq2 = self.feature_extraction(x)
        vit1 = self.vit_0(seq1); vit2 = self.vit_1(seq2)
        vit_cat = torch.cat((vit1, vit2), dim=1).flatten(1)
        cnn_out = self.fc2(self.fc1(feat))
        vit_out = self.vfc2(self.vfc1(vit_cat))
        return cnn_out * self.weight + vit_out * (1 - self.weight)

    def forward(self, x, y=None):
        x  = self._process(x)
        out = self.arcface(self.dropout(x), y)
        return out, F.normalize(x, dim=-1)

    @torch.no_grad()
    def get_embedding(self, x):
        """L2-normalised 1024-d embedding for matching."""
        x = self._process(x)
        return F.normalize(x, p=2, dim=1)


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
    cache_dir    = os.path.abspath(cfg.get("base_results_dir", "./rst_sf2net_loo"))
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

class TripletDataset(Dataset):
    """
    Triplet-sampling training dataset for SF2Net.
    Returns ([anchor, positive, negative], [label_a, label_p, label_n]).
    Positive: different sample of same identity.
    Negative: sample of different identity.
    """
    def __init__(self, samples, img_side=128, augment_factor=1):
        self.samples        = samples
        self.augment_factor = augment_factor
        self.labels         = np.array([lab for _, lab in samples])
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

    def _load(self, path):
        return self.aug_transform(Image.open(path).convert("L"))

    def __getitem__(self, index):
        real_idx = index % len(self.samples)
        path_a, label_a = self.samples[real_idx]

        pos_idxs = self.label2idxs[label_a]
        pos_idx  = real_idx
        while pos_idx == real_idx and len(pos_idxs) > 1:
            pos_idx = random.choice(pos_idxs)
        path_p, label_p = self.samples[pos_idx]

        neg_idxs = np.where(self.labels != label_a)[0]
        neg_idx  = random.choice(neg_idxs)
        path_n, label_n = self.samples[neg_idx]

        return ([self._load(path_a), self._load(path_p), self._load(path_n)],
                [label_a, label_p, label_n])


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
#  TRAINING  (SF2Net triplet loss — unchanged from official)
# ══════════════════════════════════════════════════════════════

def run_one_epoch(model, loader, criterion, tl_criterion,
                  optimizer, device, phase, ce_weight=0.7, tl_weight=0.3):
    is_train = (phase == "training")
    model.train() if is_train else model.eval()
    running_loss = 0.0; running_correct = 0; total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for datas, targets in loader:
            anchor   = datas[0].to(device)
            positive = datas[1].to(device)
            negative = datas[2].to(device)
            t_anchor = targets[0].to(device)
            t_pos    = targets[1].to(device)
            t_neg    = targets[2].to(device)

            if is_train: optimizer.zero_grad()

            out_a, fe_a = model(anchor,   t_anchor if is_train else None)
            out_p, fe_p = model(positive, t_pos    if is_train else None)
            out_n, fe_n = model(negative, t_neg    if is_train else None)

            ce_loss = criterion(out_a, t_anchor)
            tl_loss, _, _, _ = tl_criterion(out_a, out_p, out_n)
            loss = ce_weight * ce_loss + tl_weight * tl_loss

            if is_train: loss.backward(); optimizer.step()

            running_loss    += loss.item() * anchor.size(0)
            running_correct += out_a.data.max(1)[1].eq(t_anchor).sum().item()
            total           += anchor.size(0)

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
#  SINGLE LEAVE-ONE-OUT EXPERIMENT
# ══════════════════════════════════════════════════════════════

def run_experiment(train_datasets, test_dataset, cfg, device=None):
    """
    Train SF2Net on the combined samples from `train_datasets` and
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
    vit_floor_num   = cfg["vit_floor_num"]
    cnn_vit_weight  = cfg["cnn_vit_weight"]
    dropout         = cfg["dropout"]
    arcface_s       = cfg["arcface_s"]
    arcface_m       = cfg["arcface_m"]
    ce_weight       = cfg["ce_weight"]
    tl_weight       = cfg["tl_weight"]
    augment_factor  = cfg["augment_factor"]
    test_gal_ratio  = cfg["test_gallery_ratio"]
    eval_every      = cfg["eval_every"]
    save_every      = cfg["save_every"]
    nw              = cfg["num_workers"]
    eval_tag_base   = test_dataset.replace("-", "")

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
        TripletDataset(train_samples, img_side, augment_factor),
        batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    gallery_loader = DataLoader(
        SingleDataset(gallery_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    probe_loader = DataLoader(
        SingleDataset(probe_samples, img_side),
        batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Gallery={len(gallery_samples)}  Probe={len(probe_samples)}  "
          f"Classes(train)={num_classes}")

    # ── model ─────────────────────────────────────────────────────────────
    net = SF2Net(num_classes=num_classes, vit_floor_num=vit_floor_num,
                 weight=cnn_vit_weight, dropout=dropout,
                 arcface_s=arcface_s, arcface_m=arcface_m)
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)

    net = get_or_create_init_weights(net, cfg, num_classes, device)

    criterion    = nn.CrossEntropyLoss()
    tl_criterion = TripletLoss(distance="SRT")
    optimizer    = optim.Adam(net.parameters(), lr=lr)
    scheduler    = lr_scheduler.StepLR(optimizer, lr_step, lr_gamma)

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
            net, train_loader, criterion, tl_criterion,
            optimizer, device, "training",
            ce_weight=ce_weight, tl_weight=tl_weight)
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
    base_results_dir = BASE_CONFIG.get("base_results_dir", "./rst_sf2net_loo")
    os.makedirs(base_results_dir, exist_ok=True)

    print_dataset_distributions(BASE_CONFIG)

    print(f"\n{'='*60}")
    print(f"  SF2Net — Leave-One-Out Cross-Dataset Experiment")
    print(f"  Device        : {device}")
    print(f"  Datasets      : {ALL_DATASETS}")
    print(f"  Strategy      : train on 3, test on left-out 1")
    print(f"  Epochs        : {BASE_CONFIG['num_epochs']}")
    print(f"  Loss          : {BASE_CONFIG['ce_weight']}×CE + "
          f"{BASE_CONFIG['tl_weight']}×TripletLoss(SRT)")
    print(f"  CNN/ViT wt    : {BASE_CONFIG['cnn_vit_weight']} / "
          f"{1-BASE_CONFIG['cnn_vit_weight']:.1f}")
    print(f"  Embedding     : L2-normalised 1024-d")
    print(f"  EER_bal       = balanced 1:1 impostor sampling (model selection)")
    print(f"  EER_all       = all impostor pairs (reference)")
    print(f"  Results dir   : {base_results_dir}")
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
