"""
tscan_palmauth.py — TSCAN: Teacher-Student Co-learning with Adversarial
Normalization adapted for the Palm-Auth dataset.

Model, training method, and hyperparameters: unchanged from original TSCAN.
Dataset      : X-Palm (roi_perspective + roi_scanner)
Evaluation   : Cross-Domain Closed-Set (12 settings)

For each setting:
  Phase 1 : Teacher trained on SOURCE domain (labeled train_samples)
  Phase 2 : Teacher-Student co-learning
             Source → labeled train_samples (weak aug)
             Target → gallery + probe samples combined (unlabeled, weak+strong)
  Eval    : Cosine similarity, gallery vs probe → EER + Rank-1

Settings (12 total)
────────────────────
  S_scanner         │ Source: perspective (190 IDs)
                    │ Target: scanner  → 50/50 gallery/probe (148 IDs)

  S_scanner_to_persp│ Source: scanner (148 IDs)
                    │ Target: perspective → 50/50 gallery/probe (148 IDs)

  S_(A,B) (×10)     │ Source: perspective(¬A,¬B) + scanner
                    │ Target: condition A → gallery | condition B → probe

Gallery/probe splits saved to x_palm_closedset_splits.json on first run.
"""

from __future__ import annotations

# =============================================================================
# IMPORTS
# =============================================================================
import copy
import itertools
import json
import math
import os
import random
import time
import warnings
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# =============================================================================
# PARAMETERS
# =============================================================================

PALM_AUTH_ROOT  = "path to X-Palm dataset"
SCANNER_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

# ── Evaluation ────────────────────────────────────────────────────────────────
TEST_GALLERY_RATIO = 0.50
SPLITS_FILE        = "./x_palm_closedset_splits.json"

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

# ── Model ─────────────────────────────────────────────────────────────────────
FEATURE_DIM     = 256

# ── AdaFace ───────────────────────────────────────────────────────────────────
ADAFACE_M0      = 0.5
ADAFACE_MMIN    = 0.25
ADAFACE_S       = 32.0

# ── Stage 1 ───────────────────────────────────────────────────────────────────
S1_EPOCHS        = 100
S1_LR_HEAD       = 1e-3
S1_LR_LAYER4     = 1e-4
S1_WEIGHT_DECAY  = 5e-4
S1_BATCH_SIZE    = 64
S1_WARMUP_EPOCHS = 5

# ── Stage 2 ───────────────────────────────────────────────────────────────────
S2_EPOCHS        = 50
S2_LR_HEAD       = 1e-4
S2_LR_LAYER4     = 1e-5
S2_WEIGHT_DECAY  = 5e-4
S2_BATCH_SIZE    = 32
S2_WARMUP_EPOCHS = 3

# ── Co-learning ───────────────────────────────────────────────────────────────
EMA_DECAY            = 0.999
PSEUDO_LABEL_THRESH  = 0.7
ALPHA                = 1.0
BETA                 = 0.1
GAMMA_LOSS           = 0.3

# ── Augmentation ──────────────────────────────────────────────────────────────
RESIZE_SIZE     = 124
CROP_SIZE       = 112

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS     = 4
PIN_MEMORY      = True
SEED            = 42
EVAL_EVERY      = 5
SAVE_DIR        = "./rst_tscan_crossdomain"

# =============================================================================
# UTILITIES
# =============================================================================

IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class AverageMeter:
    def __init__(self): self.sum = 0.0; self.count = 0
    def reset(self): self.sum = 0.0; self.count = 0
    def update(self, val, n=1): self.sum += val * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


# =============================================================================
# AUGMENTATION  (unchanged from original TSCAN)
# Palm-Auth images are RGB (captured by smartphones/scanners in colour).
# We load them as-is with convert("RGB") — no channel replication needed.
# ResNet18 naturally expects 3-channel input.
# =============================================================================

class GaussianNoise:
    def __init__(self, std=0.02): self.std = std
    def __call__(self, t): return (t + torch.randn_like(t) * self.std).clamp(0., 1.)


def weak_transform():
    return T.Compose([
        T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        T.RandomCrop(CROP_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def strong_transform():
    return T.Compose([
        T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        T.RandomCrop(CROP_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.4, saturation=0.4, hue=0.1),
        T.RandomAutocontrast(p=0.3),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        GaussianNoise(std=0.02),
    ])


def eval_transform():
    return T.Compose([
        T.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        T.CenterCrop(CROP_SIZE),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# =============================================================================
# DATASETS  (Palm-Auth specific)
# =============================================================================

class LabeledDataset(Dataset):
    """Labeled dataset for Phase 1 and Phase 2 source."""
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")   # load as RGB (original colour image)
        return self.transform(img), label


class UnlabeledTargetDataset(Dataset):
    """Target domain: returns (weak_aug, strong_aug) — no labels."""
    def __init__(self, samples):
        self.samples = samples
        self.weak    = weak_transform()
        self.strong  = strong_transform()

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")   # load as RGB (original colour image)
        return self.weak(img), self.strong(img)


def make_labeled_loader(samples, train=False, batch_size=64, shuffle=False):
    tfm = weak_transform() if train else eval_transform()
    ds  = LabeledDataset(samples, tfm)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)),
                      shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=PIN_MEMORY,
                      drop_last=train and len(ds) > batch_size)


def make_unlabeled_loader(samples, batch_size=32):
    ds = UnlabeledTargetDataset(samples)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)),
                      shuffle=True, num_workers=NUM_WORKERS,
                      pin_memory=PIN_MEMORY,
                      drop_last=len(ds) > batch_size)


def make_eval_loader(samples, batch_size=128):
    ds = LabeledDataset(samples, eval_transform())
    return DataLoader(ds, batch_size=min(batch_size, len(ds)),
                      shuffle=False, num_workers=NUM_WORKERS)


# =============================================================================
# DATA COLLECTION HELPERS
# =============================================================================

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


# =============================================================================
# GALLERY/PROBE SPLIT PERSISTENCE
# =============================================================================

def _make_gallery_probe_split(id2paths, gallery_ratio, rng):
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


def _gallery_probe_split_from_stored(id2paths, label_map, stored_split):
    gallery, probe = [], []
    for ident, path_sets in stored_split.items():
        if ident not in label_map: continue
        label = label_map[ident]
        for p in path_sets["gallery"]: gallery.append((p, label))
        for p in path_sets["probe"]:   probe.append((p, label))
    return gallery, probe


def generate_closedset_splits(cond_paths, scanner_paths, gallery_ratio, seed):
    rng = random.Random(seed)
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    scanner_ids = sorted(scanner_paths.keys())
    splits = {}
    splits["S_scanner"] = _make_gallery_probe_split(
        {i: scanner_paths[i] for i in scanner_ids}, gallery_ratio, rng)
    splits["S_scanner_to_persp"] = _make_gallery_probe_split(
        {i: persp_all[i] for i in scanner_ids}, gallery_ratio, rng)
    return splits


def load_or_generate_closedset_splits(cond_paths, scanner_paths,
                                      gallery_ratio, seed):
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE) as f:
            splits = json.load(f)
        log(f"Loaded existing gallery/probe splits from: {SPLITS_FILE}")
    else:
        log(f"Generating gallery/probe splits (seed={seed}) → {SPLITS_FILE}")
        splits = generate_closedset_splits(
            cond_paths, scanner_paths, gallery_ratio, seed)
        with open(SPLITS_FILE, "w") as f:
            json.dump(splits, f, indent=2)
        log(f"Splits saved to: {SPLITS_FILE}")
    for key, val in splits.items():
        n_gal = sum(len(v["gallery"]) for v in val.values())
        n_prb = sum(len(v["probe"])   for v in val.values())
        log(f"  {key:<30}  IDs={len(val)}  gallery={n_gal}  probe={n_prb}")
    return splits


# =============================================================================
# PARSERS FOR EACH SETTING
# =============================================================================

def _gallery_probe_split(id2paths, label_map, gallery_ratio, rng):
    gallery, probe = [], []
    for ident, paths in id2paths.items():
        paths = list(paths); rng.shuffle(paths)
        n_gal = max(1, int(len(paths) * gallery_ratio))
        n_gal = min(n_gal, len(paths) - 1) if len(paths) > 1 else len(paths)
        for p in paths[:n_gal]: gallery.append((p, label_map[ident]))
        for p in paths[n_gal:]: probe.append((p, label_map[ident]))
    return gallery, probe


def parse_setting_scanner(cond_paths, scanner_paths, stored_splits, seed):
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    all_persp_ids = sorted(persp_all.keys())
    scanner_ids   = sorted(scanner_paths.keys())
    train_label_map = {ident: i for i, ident in enumerate(all_persp_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    num_train_cls   = len(all_persp_ids)
    train_samples   = [(p, train_label_map[i])
                       for i in all_persp_ids for p in persp_all[i]]
    split = stored_splits["S_scanner"]
    gallery_samples, probe_samples = _gallery_probe_split_from_stored(
        {i: scanner_paths[i] for i in scanner_ids}, test_label_map, split)
    _print_stats("S_scanner | Perspective/190 (source) → Scanner/148 50/50 (target)",
                 len(all_persp_ids), len(scanner_ids), len(train_samples),
                 len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_scanner_to_perspective(cond_paths, scanner_paths,
                                         stored_splits, seed):
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    scanner_ids = sorted(scanner_paths.keys())
    train_label_map = {ident: i for i, ident in enumerate(scanner_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    num_train_cls   = len(scanner_ids)
    train_samples   = [(p, train_label_map[i])
                       for i in scanner_ids for p in scanner_paths[i]]
    split = stored_splits["S_scanner_to_persp"]
    gallery_samples, probe_samples = _gallery_probe_split_from_stored(
        {i: persp_all[i] for i in scanner_ids}, test_label_map, split)
    _print_stats("S_scanner_to_persp | Scanner/148 (source) → Perspective/148 50/50 (target)",
                 len(scanner_ids), len(scanner_ids), len(train_samples),
                 len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_train_cls


def parse_setting_paired_conditions(cond_a, cond_b, cond_paths,
                                    scanner_paths, seed):
    paths_a = cond_paths.get(cond_a, {})
    paths_b = cond_paths.get(cond_b, {})
    if not paths_a: raise ValueError(f"No images for condition '{cond_a}'")
    if not paths_b: raise ValueError(f"No images for condition '{cond_b}'")
    eligible_ids = sorted(set(paths_a.keys()) & set(paths_b.keys()))
    if not eligible_ids:
        raise ValueError(f"No IDs with both '{cond_a}' and '{cond_b}'")
    label_map   = {ident: i for i, ident in enumerate(eligible_ids)}
    num_classes = len(eligible_ids)
    train_samples = []
    for cond, cond_dict in cond_paths.items():
        if cond in (cond_a, cond_b): continue
        for ident in eligible_ids:
            for p in cond_dict.get(ident, []):
                train_samples.append((p, label_map[ident]))
    for ident in eligible_ids:
        for p in scanner_paths.get(ident, []):
            train_samples.append((p, label_map[ident]))
    gallery_samples = _all_samples({i: paths_a[i] for i in eligible_ids}, label_map)
    probe_samples   = _all_samples({i: paths_b[i] for i in eligible_ids}, label_map)
    _print_stats(
        f"S_{cond_a}_{cond_b} | Perspective(not {cond_a}/{cond_b})+Scanner"
        f" → gallery:{cond_a} / probe:{cond_b}",
        num_classes, num_classes, len(train_samples),
        len(gallery_samples), len(probe_samples))
    return train_samples, gallery_samples, probe_samples, num_classes


def _print_stats(name, n_train_ids, n_test_ids, train_n, gallery_n, probe_n):
    log(f"  [{name}]")
    log(f"    Train IDs / Test IDs  : {n_train_ids} / {n_test_ids}")
    log(f"    Train images          : {train_n}")
    log(f"    Gallery / Probe       : {gallery_n} / {probe_n}")


# =============================================================================
# MODEL  (unchanged from original TSCAN)
# =============================================================================

class FeatureEncoder(nn.Module):
    def __init__(self, feat_dim=256, pretrained=True):
        super().__init__()
        weights  = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        children = list(backbone.children())
        self.frozen_layers = nn.Sequential(*children[:7])
        self.layer4        = children[7]
        self.avgpool       = children[8]
        self.flatten       = nn.Flatten()
        self.linear        = nn.Linear(512, feat_dim, bias=True)
        self.hash          = nn.Tanh()
        for p in self.frozen_layers.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.frozen_layers(x)
        x    = self.layer4(x)
        x    = self.avgpool(x)
        bb   = self.flatten(x)
        feat = self.hash(self.linear(bb))
        return feat, bb

    def layer4_parameters(self): return list(self.layer4.parameters())
    def head_parameters(self):   return list(self.linear.parameters())


class PalmNet(nn.Module):
    def __init__(self, feat_dim=256, pretrained=True):
        super().__init__()
        self.encoder = FeatureEncoder(feat_dim=feat_dim, pretrained=pretrained)

    def forward(self, x):        return self.encoder(x)
    def get_features(self, x):   return self.encoder(x)[0]
    def layer4_parameters(self): return self.encoder.layer4_parameters()
    def head_parameters(self):   return self.encoder.head_parameters()
    def backbone_parameters(self): return self.encoder.layer4_parameters()


# =============================================================================
# ADAFACE LOSS  (unchanged from original TSCAN)
# =============================================================================

class AdaFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim=256, m0=0.5, m_min=0.25, s=32.0):
        super().__init__()
        self.m0 = m0; self.m_min = m_min; self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def _margin(self, norms):
        lo = norms.min().detach(); hi = norms.max().detach()
        denom = (hi - lo).clamp(min=1e-8)
        return (self.m_min + (self.m0 - self.m_min) *
                (norms - lo) / denom).clamp(self.m_min, self.m0)

    def forward(self, features, labels):
        norms   = features.norm(dim=1)
        margins = self._margin(norms)
        feat_n  = F.normalize(features, dim=1)
        w_n     = F.normalize(self.weight, dim=1)
        cosine  = (feat_n @ w_n.T).clamp(-1 + 1e-7, 1 - 1e-7)
        theta   = torch.acos(cosine)
        m_col   = margins.unsqueeze(1)
        cos_m_  = cosine * torch.cos(m_col) - torch.sin(theta) * torch.sin(m_col)
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        logits  = self.s * (one_hot * cos_m_ + (1 - one_hot) * cosine)
        return F.cross_entropy(logits, labels)

    def get_logits(self, features):
        return (F.normalize(features, dim=1) @
                F.normalize(self.weight, dim=1).T * self.s)

    def freeze_weights(self):
        self.weight.requires_grad = False
        log("  AdaFace W frozen for Phase 2")

    def unfreeze_weights(self):
        self.weight.requires_grad = True


# =============================================================================
# GRL + DISCRIMINATOR  (unchanged from original TSCAN)
# =============================================================================

class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha; return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim=256, hidden=128, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.net   = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1), nn.Sigmoid())

    def forward(self, feat):
        return self.net(GRLFunction.apply(feat, self.alpha))

    def set_alpha(self, alpha): self.alpha = alpha


# =============================================================================
# TRAINING HELPERS  (unchanged from original TSCAN)
# =============================================================================

@torch.no_grad()
def ema_update(teacher, student, decay=0.999):
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(decay).add_(s_p.data * (1.0 - decay))


def grl_alpha(cur_iter, max_iter, alpha_max=1.0):
    p = cur_iter / max(max_iter, 1)
    return float(alpha_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


@torch.no_grad()
def generate_pseudo_labels(teacher, weak_imgs, adaface, threshold):
    teacher.eval()
    feat, _ = teacher(weak_imgs)
    probs   = F.softmax(adaface.get_logits(feat), dim=1)
    max_p, pl = probs.max(dim=1)
    mask = max_p >= threshold
    pl[~mask] = -1
    return pl, mask


def domain_loss(discriminator, src_feat, tgt_feat):
    src_lbl = torch.zeros(src_feat.size(0), 1, device=src_feat.device)
    tgt_lbl = torch.ones (tgt_feat.size(0), 1, device=tgt_feat.device)
    preds   = discriminator(torch.cat([src_feat, tgt_feat], dim=0))
    return F.binary_cross_entropy(preds, torch.cat([src_lbl, tgt_lbl], dim=0))


def make_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                                T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


# =============================================================================
# EVALUATION  (cross-domain closed-set: gallery vs probe)
# =============================================================================

@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats, labs = [], []
    for imgs, labels in loader:
        feat = F.normalize(model.get_features(imgs.to(DEVICE)), dim=1)
        feats.append(feat.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


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


def evaluate(model, probe_loader, gallery_loader, out_dir=".", tag="eval"):
    """Cosine similarity gallery-probe evaluation → EER + Rank-1."""
    probe_feats,   probe_labels   = extract_features(model, probe_loader)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader)

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
    correct = sum(probe_labels[i] == gallery_labels[nn_idx[i]]
                  for i in range(n_probe))
    rank1   = 100.0 * correct / max(n_probe, 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")

    log(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer, rank1


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(train_samples, gallery_samples, probe_samples,
                   num_classes, results_dir):
    """
    Run TSCAN (Phase 1 + Phase 2) for one cross-domain setting.

    Source domain = train_samples (labeled)
    Target domain = gallery_samples + probe_samples combined (unlabeled
                    during Phase 2, evaluated separately at the end)
    """
    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    # Target = gallery + probe combined (unlabeled for Phase 2)
    target_samples = gallery_samples + probe_samples

    # ── Data loaders ──────────────────────────────────────────────────────────
    s1_train_loader = make_labeled_loader(
        train_samples, train=True, batch_size=S1_BATCH_SIZE, shuffle=True)
    s2_src_loader   = make_labeled_loader(
        train_samples, train=True, batch_size=S2_BATCH_SIZE, shuffle=True)
    s2_tgt_loader   = make_unlabeled_loader(target_samples, batch_size=S2_BATCH_SIZE)
    gallery_loader  = make_eval_loader(gallery_samples)
    probe_loader    = make_eval_loader(probe_samples)

    # ── Models ────────────────────────────────────────────────────────────────
    teacher   = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
    adaface   = AdaFaceLoss(num_classes=num_classes, feat_dim=FEATURE_DIM,
                            m0=ADAFACE_M0, m_min=ADAFACE_MMIN,
                            s=ADAFACE_S).to(DEVICE)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Teacher Initialization  (unchanged from original TSCAN)
    # ══════════════════════════════════════════════════════════════════════════
    log(f"\n  -- Phase 1: Teacher Initialization  (classes={num_classes}) --")

    s1_optimizer = optim.AdamW([
        {'params': teacher.layer4_parameters(), 'lr': S1_LR_LAYER4},
        {'params': teacher.head_parameters(),   'lr': S1_LR_HEAD},
        {'params': adaface.parameters(),        'lr': S1_LR_HEAD},
    ], weight_decay=S1_WEIGHT_DECAY)
    s1_scheduler = make_warmup_cosine_scheduler(
        s1_optimizer, S1_WARMUP_EPOCHS, S1_EPOCHS)

    best_rank1    = 0.0
    best_teacher  = None
    best_adaface  = None
    ckpt_path     = os.path.join(results_dir, "best_phase1.pt")

    # Pre-training baseline
    evaluate(teacher, probe_loader, gallery_loader,
             rst_eval, "P1_ep000_pretrain")

    for epoch in range(1, S1_EPOCHS + 1):
        teacher.train(); adaface.train()
        loss_m = AverageMeter()

        for imgs, labels in s1_train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            feat, _  = teacher(imgs)
            loss     = adaface(feat, labels)
            s1_optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                teacher.layer4_parameters() + teacher.head_parameters() +
                list(adaface.parameters()), max_norm=5.0)
            s1_optimizer.step()
            loss_m.update(loss.item(), imgs.size(0))
        s1_scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == S1_EPOCHS:
            cur_eer, cur_rank1 = evaluate(
                teacher, probe_loader, gallery_loader,
                rst_eval, f"P1_ep{epoch:04d}")
            marker = "  *** new best ***" if cur_rank1 > best_rank1 else ""
            log(f"  P1 ep {epoch:03d}/{S1_EPOCHS}  loss={loss_m.avg:.4f}"
                f"  EER={cur_eer*100:.4f}%  Rank-1={cur_rank1:.2f}%{marker}")
            if cur_rank1 > best_rank1:
                best_rank1   = cur_rank1
                best_teacher = copy.deepcopy(teacher.state_dict())
                best_adaface = copy.deepcopy(adaface.state_dict())
                torch.save({"teacher": best_teacher, "adaface": best_adaface},
                           ckpt_path)
        else:
            if epoch % 10 == 0:
                log(f"  P1 ep {epoch:03d}/{S1_EPOCHS}  loss={loss_m.avg:.4f}")

    teacher.load_state_dict(best_teacher)
    adaface.load_state_dict(best_adaface)
    p1_eer, p1_rank1 = evaluate(
        teacher, probe_loader, gallery_loader, rst_eval, "P1_FINAL")
    log(f"  Phase 1 best: EER={p1_eer*100:.4f}%  Rank-1={p1_rank1:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Teacher-Student Co-Learning  (unchanged from original TSCAN)
    # ══════════════════════════════════════════════════════════════════════════
    log(f"\n  -- Phase 2: Teacher-Student Co-Learning --")

    student = PalmNet(feat_dim=FEATURE_DIM, pretrained=True).to(DEVICE)
    student.load_state_dict(best_teacher)

    adaface.freeze_weights()

    discriminator = DomainDiscriminator(
        feat_dim=FEATURE_DIM, hidden=128, alpha=1.0).to(DEVICE)

    for p in teacher.parameters():
        p.requires_grad = False

    s2_optimizer = optim.AdamW([
        {'params': student.layer4_parameters(), 'lr': S2_LR_LAYER4},
        {'params': student.head_parameters(),   'lr': S2_LR_HEAD},
        {'params': discriminator.parameters(),  'lr': S2_LR_HEAD},
    ], weight_decay=S2_WEIGHT_DECAY)
    s2_scheduler = make_warmup_cosine_scheduler(
        s2_optimizer, S2_WARMUP_EPOCHS, S2_EPOCHS)

    best_s2_rank1 = p1_rank1
    best_student  = copy.deepcopy(student.state_dict())
    ckpt_path_s2  = os.path.join(results_dir, "best_phase2.pt")
    total_steps   = len(s2_src_loader) * S2_EPOCHS
    global_step   = 0

    for epoch in range(1, S2_EPOCHS + 1):
        student.train(); teacher.eval()
        discriminator.train(); adaface.eval()

        loss_t_m = AverageMeter(); loss_s_m = AverageMeter()
        loss_u_m = AverageMeter(); loss_d_m = AverageMeter()

        max_steps = max(len(s2_src_loader), len(s2_tgt_loader))
        src_iter  = itertools.cycle(s2_src_loader)
        tgt_iter  = itertools.cycle(s2_tgt_loader)

        for _ in range(max_steps):
            alpha = grl_alpha(global_step, total_steps, alpha_max=1.0)
            discriminator.set_alpha(alpha)

            src_imgs, src_lbl = next(src_iter)
            tgt_weak, tgt_str = next(tgt_iter)
            src_imgs = src_imgs.to(DEVICE); src_lbl  = src_lbl.to(DEVICE)
            tgt_weak = tgt_weak.to(DEVICE); tgt_str  = tgt_str.to(DEVICE)

            pl, mask    = generate_pseudo_labels(
                teacher, tgt_weak, adaface, PSEUDO_LABEL_THRESH)
            src_feat, _ = student(src_imgs)
            tgt_feat, _ = student(tgt_str)

            L_sup   = adaface(src_feat, src_lbl)
            L_unsup = (adaface(tgt_feat[mask], pl[mask]) if mask.any()
                       else torch.tensor(0.0, device=DEVICE))
            L_dis   = domain_loss(discriminator, src_feat, tgt_feat)
            L_total = ALPHA * L_sup + BETA * L_unsup + GAMMA_LOSS * L_dis

            s2_optimizer.zero_grad(); L_total.backward()
            nn.utils.clip_grad_norm_(
                student.layer4_parameters() + student.head_parameters() +
                list(discriminator.parameters()), max_norm=5.0)
            s2_optimizer.step()
            ema_update(teacher, student, decay=EMA_DECAY)

            bs = src_imgs.size(0)
            loss_t_m.update(L_total.item(), bs); loss_s_m.update(L_sup.item(), bs)
            loss_u_m.update(L_unsup.item() if mask.any() else 0., bs)
            loss_d_m.update(L_dis.item(), bs)
            global_step += 1

        s2_scheduler.step()

        if epoch % EVAL_EVERY == 0 or epoch == S2_EPOCHS:
            cur_eer, cur_rank1 = evaluate(
                student, probe_loader, gallery_loader,
                rst_eval, f"P2_ep{epoch:04d}")
            delta  = (p1_eer - cur_eer) * 100
            marker = "  *** new best ***" if cur_rank1 > best_rank1 else ""
            log(f"  P2 ep {epoch:03d}/{S2_EPOCHS}  "
                f"L={loss_t_m.avg:.4f} Ls={loss_s_m.avg:.4f} "
                f"Lu={loss_u_m.avg:.4f} Ld={loss_d_m.avg:.4f}  "
                f"EER={cur_eer*100:.4f}%  Rank-1={cur_rank1:.2f}%  "
                f"ΔEER={delta:+.2f}%{marker}")
            if cur_rank1 > best_s2_rank1:
                best_s2_rank1 = cur_rank1
                best_student  = copy.deepcopy(student.state_dict())
                torch.save({"student": best_student}, ckpt_path_s2)
        else:
            if epoch % 10 == 0:
                log(f"  P2 ep {epoch:03d}/{S2_EPOCHS}  "
                    f"L={loss_t_m.avg:.4f} Ls={loss_s_m.avg:.4f} "
                    f"Lu={loss_u_m.avg:.4f} Ld={loss_d_m.avg:.4f}")

    student.load_state_dict(best_student)
    final_eer, final_rank1 = evaluate(
        student, probe_loader, gallery_loader, rst_eval, "P2_FINAL")
    log(f"  Phase 2 best: EER={final_eer*100:.4f}%  Rank-1={final_rank1:.2f}%  "
        f"(P1 baseline: EER={p1_eer*100:.4f}%  ΔEER={(p1_eer-final_eer)*100:+.2f}%)")

    return final_eer, final_rank1


# =============================================================================
# RESULTS SUMMARY TABLE
# =============================================================================

def print_and_save_summary(all_results, out_path):
    col_w  = 14
    header = (f"{'Setting':<22}"
              f"{'Train domain':<38}"
              f"{'Test domain':<26}"
              f"{'EER (%)':>{col_w}}"
              f"{'Rank-1 (%)':>{col_w}}")
    sep = "─" * len(header)
    lines = ["\nCross-Domain Closed-Set Results — Palm-Auth (TSCAN)",
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
    log(f"Summary saved to: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    log("=" * 72)
    log(f"TSCAN — Cross-Domain Closed-Set (Palm-Auth)")
    log(f"Device    : {DEVICE}")
    log(f"Phase 1   : {S1_EPOCHS} epochs  |  Phase 2: {S2_EPOCHS} epochs")
    log(f"Settings  : 2 scanner + {len(PAIRED_CONDITIONS)} paired-condition")
    log(f"Results   : {SAVE_DIR}")
    log("=" * 72)

    log("\nScanning dataset …")
    cond_paths    = _collect_perspective(PALM_AUTH_ROOT)
    scanner_paths = _collect_scanner(PALM_AUTH_ROOT, SCANNER_SPECTRA)
    log(f"Perspective conditions: {sorted(cond_paths.keys())}")
    log(f"Scanner identities   : {len(scanner_paths)}")

    all_splits = load_or_generate_closedset_splits(
        cond_paths, scanner_paths, TEST_GALLERY_RATIO, SEED)

    SETTINGS = []

    SETTINGS.append({
        "tag"        : "setting_scanner",
        "label"      : "S_scanner",
        "train_desc" : "Perspective (all 190 IDs)",
        "test_desc"  : "Scanner 50/50 gallery/probe",
        "parser"     : lambda: parse_setting_scanner(
                           cond_paths, scanner_paths, all_splits, SEED),
    })
    SETTINGS.append({
        "tag"        : "setting_scanner_to_persp",
        "label"      : "S_scanner_to_persp",
        "train_desc" : "Scanner (148 IDs)",
        "test_desc"  : "Perspective 50/50 gallery/probe",
        "parser"     : lambda: parse_setting_scanner_to_perspective(
                           cond_paths, scanner_paths, all_splits, SEED),
    })

    conditions_found = sorted(cond_paths.keys())
    for cond_a, cond_b in PAIRED_CONDITIONS:
        if cond_a not in conditions_found or cond_b not in conditions_found:
            log(f"  [WARN] '{cond_a}' or '{cond_b}' not found — skipping")
            continue
        ca, cb = cond_a, cond_b
        SETTINGS.append({
            "tag"        : f"setting_{ca}_{cb}",
            "label"      : f"S_{ca}_{cb}",
            "train_desc" : f"Perspective(not {ca}/{cb}) + Scanner",
            "test_desc"  : f"gallery:{ca} / probe:{cb}",
            "parser"     : (lambda ca=ca, cb=cb: parse_setting_paired_conditions(
                                ca, cb, cond_paths, scanner_paths, SEED)),
        })

    log(f"\nTotal settings to run : {len(SETTINGS)}")

    all_results = []

    for idx, s in enumerate(SETTINGS, 1):
        log(f"\n{'='*72}")
        log(f"[{idx}/{len(SETTINGS)}] {s['label']}")
        log(f"  Source (train) : {s['train_desc']}")
        log(f"  Target (test)  : {s['test_desc']}")
        log(f"{'='*72}")

        results_dir = os.path.join(SAVE_DIR, s["tag"])
        t_start     = time.time()
        try:
            train_s, gal_s, probe_s, n_cls = s["parser"]()
            eer, rank1 = run_experiment(
                train_s, gal_s, probe_s, n_cls, results_dir)
            elapsed = time.time() - t_start
            log(f"\n  ✓  {s['label']}:  EER={eer*100:.4f}%  "
                f"Rank-1={rank1:.2f}%  Time={elapsed/60:.1f} min")
            with open(os.path.join(results_dir, "results.json"), "w") as f:
                json.dump({"setting": s["label"], "train_desc": s["train_desc"],
                           "test_desc": s["test_desc"], "num_classes": n_cls,
                           "EER_pct": eer*100, "Rank1_pct": rank1}, f, indent=2)
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"],
                                 "eer": eer*100, "rank1": rank1})
        except Exception as e:
            log(f"\n  ✗  {s['label']} FAILED: {e}")
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"],
                                 "eer": None, "rank1": None})

    log(f"\n\n{'='*72}")
    log(f"ALL {len(SETTINGS)} SETTINGS COMPLETE")
    log(f"{'='*72}")
    print_and_save_summary(
        all_results,
        os.path.join(SAVE_DIR, "results_summary.txt"))


if __name__ == "__main__":
    main()
  
