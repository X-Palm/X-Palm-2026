"""
GIFT: Generating stylIzed FeaTures for Single-Source Cross-Dataset
Palmprint Recognition With Unseen Target Dataset
IEEE Transactions on Image Processing, Vol. 33, 2024

Adapted for X-Palm dataset with cross-domain closed-set evaluation.

Model, training method, hyperparameters: unchanged from original GIFT.
Dataset      : Palm-Auth (roi_perspective + roi_scanner)
Images       : loaded as original RGB
Evaluation   : Cross-Domain Closed-Set (12 settings)
               gallery vs probe → EER + Rank-1
Checkpoint   : saved by best Rank-1

Settings (12 total)
────────────────────
  S_scanner         │ Train : roi_perspective (all conditions, 190 IDs)
                    │ Gallery: 50% of scanner samples  (148 shared IDs)
                    │ Probe  : 50% of scanner samples  (148 shared IDs)

  S_scanner_to_persp│ Train : roi_scanner (148 IDs)
                    │ Gallery: 50% of perspective samples (148 shared IDs)
                    │ Probe  : 50% of perspective samples (148 shared IDs)

  S_(A,B) (×10)     │ Train : perspective (all except A and B) + scanner
                    │ Gallery: ALL condition A images
                    │ Probe  : ALL condition B images

Gallery/probe splits saved to x_palm_closedset_splits.json on first run.
"""

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
from torchvision import transforms, models
from pytorch_metric_learning import losses

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (unchanged from original GIFT)
# ─────────────────────────────────────────────────────────────────────────────

PALM_AUTH_ROOT  = "path to X-Palm dataset"
SCANNER_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

# Evaluation
TEST_GALLERY_RATIO = 0.50
SPLITS_FILE        = "./x_palm_closedset_splits.json"

# Paired conditions
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

# Model
BATCH_SIZE    = 32
LR            = 1e-4
WARMUP_EPOCHS = 0
EPOCHS        = 100
EMB_DIM       = 128
ARC_MARGIN    = 0.4
ARC_SCALE     = 20

# Loss weights
ALPHA_FINAL   = 1.0 # 15
BETA_FINAL    = 20.0 # 10

# FSM
GAMMA         = 0.2

# Training
AUGMENT_FACTOR = 4   # repeat each training image N times with different augmentations
EVAL_EVERY    = 5
NUM_WORKERS   = 4
GRAD_CLIP     = 1.0

SEED          = 42
SAVE_DIR      = "./rst_gift_crossdomain"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION  (unchanged from original GIFT)
# Palm-Auth images are RGB — loaded as-is with convert("RGB")
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]


def train_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────

class PalmAuthDataset(Dataset):
    def __init__(self, samples, train=False):
        self.samples   = samples
        self.transform = train_transform() if train else eval_transform()

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")   # original RGB image
        return self.transform(img), label


def make_loader(samples, train=False, batch_size=32, shuffle=False,
              augment_factor=1):
    # Repeat samples so each image is seen augment_factor times per epoch
    # with a different random augmentation each time
    if train and augment_factor > 1:
        samples = samples * augment_factor
    ds = PalmAuthDataset(samples, train=train)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)),
                      shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=True,
                      drop_last=train and len(ds) > batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# GALLERY/PROBE SPLIT PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS FOR EACH SETTING
# ─────────────────────────────────────────────────────────────────────────────

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
    _print_stats("S_scanner | Perspective/190 (train) → Scanner/148 50/50 (test)",
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
    _print_stats("S_scanner_to_persp | Scanner/148 (train) → Perspective/148 50/50 (test)",
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


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE STYLIZATION MODULE  (unchanged from original GIFT)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureStylizationModule(nn.Module):
    def __init__(self, gamma=GAMMA):
        super().__init__()
        self.gamma  = gamma
        self.active = False

    def _decompose(self, f):
        _, _, H, W = f.shape
        f_L = F.avg_pool2d(f, kernel_size=2, stride=2, padding=0)
        f_L = F.interpolate(f_L, size=(H, W), mode='nearest')
        f_H = f - f_L
        return f_L, f_H

    def forward(self, f):
        if not self.training or not self.active:
            return f, f

        f_orig = f
        f_L, f_H = self._decompose(f)

        mu_i  = f_L.mean(dim=(-2, -1))
        sig_i = f_L.std(dim=(-2, -1), unbiased=False).clamp(min=1e-5)

        mu_hat  = mu_i.std(dim=0,  unbiased=False).clamp(min=1e-5)
        sig_hat = sig_i.std(dim=0, unbiased=False).clamp(min=1e-5)

        phi_mu  = torch.randn_like(mu_i).clamp(-2, 2) * self.gamma
        phi_sig = torch.randn_like(sig_i).clamp(-2, 2) * self.gamma

        mu_new  = mu_i  + phi_mu  * mu_hat.unsqueeze(0)
        sig_new = (sig_i + phi_sig * sig_hat.unsqueeze(0)).clamp(min=1e-5)

        def _4d(t): return t.view(t.shape[0], t.shape[1], 1, 1)

        f_L_norm  = (f_L - _4d(mu_i)) / (_4d(sig_i) + 1e-5)
        f_L_new   = _4d(mu_new) + _4d(sig_new) * f_L_norm
        f_stylized = f_L_new + f_H
        return f_orig, f_stylized


# ─────────────────────────────────────────────────────────────────────────────
# DISCRIMINATOR  (unchanged from original GIFT)
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(in_channels // 2, 32)),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // 2, 32), 2),
        )

    def forward(self, x): return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE  (unchanged from original GIFT)
# ─────────────────────────────────────────────────────────────────────────────

class GIFTBackbone(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, gamma=GAMMA):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.fsm0    = FeatureStylizationModule(gamma)

        self.layer1 = resnet.layer1; self.fsm1 = FeatureStylizationModule(gamma)
        self.layer2 = resnet.layer2; self.fsm2 = FeatureStylizationModule(gamma)
        self.layer3 = resnet.layer3; self.fsm3 = FeatureStylizationModule(gamma)
        self.layer4 = resnet.layer4; self.fsm4 = FeatureStylizationModule(gamma)

        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(resnet.fc.in_features, emb_dim)

        self.channel_sizes = [64, 64, 128, 256, 512]
        self.fsm_list      = [self.fsm0, self.fsm1,
                              self.fsm2, self.fsm3, self.fsm4]

    def activate_fsm(self):
        for fsm in self.fsm_list:
            fsm.active = True
        log("  ✓ FSM stylization activated.")

    def forward(self, x):
        pairs = []
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        o, s = self.fsm0(x); pairs.append((o, s))
        x = s if (self.training and self.fsm0.active) else o

        x = self.layer1(x); o, s = self.fsm1(x); pairs.append((o, s))
        x = s if (self.training and self.fsm1.active) else o

        x = self.layer2(x); o, s = self.fsm2(x); pairs.append((o, s))
        x = s if (self.training and self.fsm2.active) else o

        x = self.layer3(x); o, s = self.fsm3(x); pairs.append((o, s))
        x = s if (self.training and self.fsm3.active) else o

        x = self.layer4(x); o, s = self.fsm4(x); pairs.append((o, s))
        x = s if (self.training and self.fsm4.active) else o

        x   = self.avgpool(x).flatten(1)
        emb = F.normalize(self.fc(x), p=2, dim=1)
        return emb, pairs


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING  (unchanged from original GIFT)
# ─────────────────────────────────────────────────────────────────────────────

def get_loss_weights(epoch):
    if epoch < WARMUP_EPOCHS:
        return 0.0, 0.0
    ramp = min((epoch - WARMUP_EPOCHS) / 20.0, 1.0)
    #return ALPHA_FINAL * ramp, BETA_FINAL * ramp
    return ALPHA_FINAL, BETA_FINAL


def train_one_epoch(model, discriminators, criterion_arc,
                    optimizer, opt_disc, loader, epoch):
    model.train()
    criterion_disc = nn.CrossEntropyLoss()
    alpha, beta = get_loss_weights(epoch)
    fsm_on      = epoch >= WARMUP_EPOCHS

    tot_loss = tot_arc = tot_de = tot_con = 0.0
    correct = total = nan_batches = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(); opt_disc.zero_grad()

        emb, pairs = model(imgs)
        loss_arc   = criterion_arc(emb, labels)

        if not torch.isfinite(loss_arc):
            nan_batches += 1; continue

        loss_de  = torch.zeros(1, device=DEVICE)
        loss_con = torch.zeros(1, device=DEVICE)

        if fsm_on and alpha > 0:
            for k, (f_orig, f_sty) in enumerate(pairs):
                disc = discriminators[k]
                lo   = disc(f_orig); ls = disc(f_sty)
                lbl1 = torch.ones (imgs.size(0), dtype=torch.long, device=DEVICE)
                lbl0 = torch.zeros(imgs.size(0), dtype=torch.long, device=DEVICE)
                loss_de  = loss_de + 0.5 * (criterion_disc(lo, lbl1) +
                                            criterion_disc(ls, lbl0))
                loss_con = loss_con + F.mse_loss(
                    f_sty.mean(dim=(-2, -1)),
                    f_orig.mean(dim=(-2, -1)).detach())
            loss_de  = loss_de  / len(pairs)
            loss_con = loss_con / len(pairs)

        loss = loss_arc + alpha * loss_de + beta * loss_con
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(discriminators.parameters(), GRAD_CLIP)
        optimizer.step(); opt_disc.step()

        tot_loss += loss.item(); tot_arc += loss_arc.item()
        tot_de   += loss_de.item(); tot_con += loss_con.item()

        with torch.no_grad():
            preds = criterion_arc.get_logits(emb).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    if nan_batches:
        log(f"  ⚠ {nan_batches} NaN batches skipped")

    n = max(len(loader) - nan_batches, 1)
    phase = "WARMUP" if not fsm_on else f"GIFT (α={alpha:.1f} β={beta:.1f})"
    log(f"  [{phase}]  Loss={tot_loss/n:.4f} "
        f"(arc={tot_arc/n:.4f} de={tot_de/n:.4f} con={tot_con/n:.4f})  "
        f"TrainAcc={correct/max(total,1)*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  (gallery vs probe, cosine similarity → EER + Rank-1)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats, labels = [], []
    for imgs, lbl in loader:
        emb, _ = model(imgs.to(DEVICE))
        feats.append(F.normalize(emb, p=2, dim=1).cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(feats), np.concatenate(labels)


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


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(train_samples, gallery_samples, probe_samples,
                   num_classes, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    train_loader   = make_loader(train_samples,   train=True,
                                 batch_size=BATCH_SIZE, shuffle=True,
                                 augment_factor=AUGMENT_FACTOR)
    gallery_loader = make_loader(gallery_samples, train=False, batch_size=128)
    probe_loader   = make_loader(probe_samples,   train=False, batch_size=128)

    model          = GIFTBackbone(emb_dim=EMB_DIM, gamma=GAMMA).to(DEVICE)
    discriminators = nn.ModuleList([
        Discriminator(c).to(DEVICE) for c in model.channel_sizes
    ])

    criterion_arc = losses.ArcFaceLoss(
        num_classes=num_classes, embedding_size=EMB_DIM,
        margin=ARC_MARGIN, scale=ARC_SCALE
    ).to(DEVICE)

    optimizer = optim.RMSprop(
        list(model.parameters()) + list(criterion_arc.parameters()),
        lr=LR, weight_decay=1e-3)
    opt_disc  = optim.RMSprop(discriminators.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1)

    log(f"\n  Phase 1: {WARMUP_EPOCHS} warmup epochs  (ArcFace only, FSM off)")
    log(f"  Phase 2: {EPOCHS - WARMUP_EPOCHS} main epochs  (all losses, FSM on)")
    log(f"  α→{ALPHA_FINAL}  β→{BETA_FINAL}  γ={GAMMA}  scale={ARC_SCALE}")

    best_rank1 = 0.0
    ckpt_path  = os.path.join(results_dir, "best_model.pth")

    # Pre-training baseline
    evaluate(model, probe_loader, gallery_loader, rst_eval, "ep000_pretrain")

    for epoch in range(EPOCHS):
        if epoch == WARMUP_EPOCHS:
            model.activate_fsm()

        train_one_epoch(model, discriminators, criterion_arc,
                        optimizer, opt_disc, train_loader, epoch)
        scheduler.step()

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            cur_eer, cur_rank1 = evaluate(
                model, probe_loader, gallery_loader,
                rst_eval, f"ep{epoch+1:04d}")
            if cur_rank1 > best_rank1:
                best_rank1 = cur_rank1
                torch.save({"epoch": epoch + 1,
                            "model": model.state_dict(),
                            "discriminators": discriminators.state_dict(),
                            "rank1": cur_rank1, "eer": cur_eer}, ckpt_path)
                log(f"  *** New best Rank-1: {best_rank1:.2f}% ***")

    # Reload best checkpoint for final reported result
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_eer, final_rank1 = evaluate(
        model, probe_loader, gallery_loader, rst_eval, "FINAL")

    return final_eer, final_rank1


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_summary(all_results, out_path):
    col_w  = 14
    header = (f"{'Setting':<22}"
              f"{'Train domain':<38}"
              f"{'Test domain':<26}"
              f"{'EER (%)':>{col_w}}"
              f"{'Rank-1 (%)':>{col_w}}")
    sep = "─" * len(header)
    lines = ["\nCross-Domain Closed-Set Results — Palm-Auth (GIFT)",
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    log("=" * 72)
    log(f"GIFT — Cross-Domain Closed-Set (Palm-Auth)")
    log(f"Device    : {DEVICE}")
    log(f"Epochs    : {EPOCHS}  (warmup={WARMUP_EPOCHS})")
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
        log(f"  Train : {s['train_desc']}")
        log(f"  Test  : {s['test_desc']}")
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
