"""
DINOv2 ViT-S/14 — Cross-Domain Closed-Set Evaluation on X-Palm
==================================================================
Architecture : DINOv2 ViT-S/14 (pretrained, facebookresearch)
               Frozen : patch embed + transformer blocks 0-9
               Trainable : blocks 10-11 (last 2 of 12)
Loss         : ArcFace  +  λ · SupConLoss
Images       : original RGB  (ConvNeXt expects 3-channel input)
Evaluation   : gallery vs probe → EER + Rank-1
Checkpoint   : saved by best Rank-1

Settings (12 total)
────────────────────
  S_scanner         │ Train : perspective (all, 190 IDs)
                    │ Gallery: 50% scanner (148 IDs)
                    │ Probe  : 50% scanner (148 IDs)

  S_scanner_to_persp│ Train : scanner (148 IDs)
                    │ Gallery: 50% perspective (148 IDs)
                    │ Probe  : 50% perspective (148 IDs)

  S_(A,B) (×10)     │ Train : perspective(¬A,¬B) + scanner
                    │ Gallery: ALL condition A images
                    │ Probe  : ALL condition B images

Gallery/probe splits saved to x_palm_closedset_splits.json on first run.
"""

import json
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
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses as pml_losses

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (unchanged from CASIA-MS version)
# ─────────────────────────────────────────────────────────────────────────────

PALM_AUTH_ROOT  = "path to X-Palm dataset"
SCANNER_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

TEST_GALLERY_RATIO = 0.50
SPLITS_FILE        = "./x_palm_closedset_splits.json"
SAVE_DIR           = "./rst_dino_crossdomain"

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

# Training
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
LAMB         = 0.2      # SupCon weight
MARGIN       = 0.3      # ArcFace margin
SCALE        = 16       # ArcFace scale
EVAL_EVERY   = 5
NUM_WORKERS  = 4
IMG_SIZE     = 224
SEED         = 42

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION  (CompNet-style, adapted for RGB + ImageNet normalisation)
# ─────────────────────────────────────────────────────────────────────────────

base_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def _make_aug():
    """CompNet-style augmentation — one random transform picked each call."""
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.15, p=1),
            transforms.RandomChoice([
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False,
                                          center=(int(0.5*IMG_SIZE), 0)),
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False,
                                          center=(0, int(0.5*IMG_SIZE))),
            ]),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────

class TrainDataset(Dataset):
    """
    Returns (img_orig, aug1, aug2, aug3, label) — 4× augmentation per image.
    img_orig : base transform only (no augmentation)
    aug1-3   : three independently sampled CompNet-style augmentations
    In the training loop these are stacked into a 4× batch:
      [orig | aug1 | aug2 | aug3]
    so each original image is seen 4 times per epoch with different augmentations.
    """
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return (base_transform(img),
                _make_aug()(img),
                _make_aug()(img),
                _make_aug()(img),
                label)


class EvalDataset(Dataset):
    """Returns (img, label) for gallery/probe evaluation."""
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return base_transform(Image.open(path).convert("RGB")), label


def make_train_loader(samples):
    return DataLoader(TrainDataset(samples),
                      batch_size=min(BATCH_SIZE, len(samples)),
                      shuffle=True, num_workers=NUM_WORKERS,
                      pin_memory=True, drop_last=len(samples) > BATCH_SIZE)


def make_eval_loader(samples):
    return DataLoader(EvalDataset(samples),
                      batch_size=min(128, len(samples)),
                      shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _collect_perspective(data_root):
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
        n_gal = min(n_gal, len(paths) - 1) if len(paths) > 1 else n_gal
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
    return {
        "S_scanner": _make_gallery_probe_split(
            {i: scanner_paths[i] for i in scanner_ids}, gallery_ratio, rng),
        "S_scanner_to_persp": _make_gallery_probe_split(
            {i: persp_all[i] for i in scanner_ids}, gallery_ratio, rng),
    }


def load_or_generate_closedset_splits(cond_paths, scanner_paths, gallery_ratio, seed):
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE) as f: splits = json.load(f)
        log(f"Loaded existing gallery/probe splits from: {SPLITS_FILE}")
    else:
        log(f"Generating gallery/probe splits (seed={seed}) → {SPLITS_FILE}")
        splits = generate_closedset_splits(cond_paths, scanner_paths, gallery_ratio, seed)
        with open(SPLITS_FILE, "w") as f: json.dump(splits, f, indent=2)
        log(f"Splits saved to: {SPLITS_FILE}")
    for key, val in splits.items():
        n_gal = sum(len(v["gallery"]) for v in val.values())
        n_prb = sum(len(v["probe"])   for v in val.values())
        log(f"  {key:<30}  IDs={len(val)}  gallery={n_gal}  probe={n_prb}")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS — return (train_samples, gallery_samples, probe_samples, num_classes)
# ─────────────────────────────────────────────────────────────────────────────

def parse_setting_scanner(cond_paths, scanner_paths, stored_splits, seed):
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    all_persp_ids   = sorted(persp_all.keys())
    scanner_ids     = sorted(scanner_paths.keys())
    train_label_map = {ident: i for i, ident in enumerate(all_persp_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    train_samples   = [(p, train_label_map[i])
                       for i in all_persp_ids for p in persp_all[i]]
    gallery, probe  = _gallery_probe_split_from_stored(
        {i: scanner_paths[i] for i in scanner_ids}, test_label_map,
        stored_splits["S_scanner"])
    _print_stats("S_scanner", len(all_persp_ids), len(scanner_ids),
                 len(train_samples), len(gallery), len(probe))
    return train_samples, gallery, probe, len(all_persp_ids)


def parse_setting_scanner_to_perspective(cond_paths, scanner_paths, stored_splits, seed):
    persp_all = defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident, paths in cond_dict.items():
            persp_all[ident].extend(paths)
    scanner_ids     = sorted(scanner_paths.keys())
    train_label_map = {ident: i for i, ident in enumerate(scanner_ids)}
    test_label_map  = {ident: i for i, ident in enumerate(scanner_ids)}
    train_samples   = [(p, train_label_map[i])
                       for i in scanner_ids for p in scanner_paths[i]]
    gallery, probe  = _gallery_probe_split_from_stored(
        {i: persp_all[i] for i in scanner_ids}, test_label_map,
        stored_splits["S_scanner_to_persp"])
    _print_stats("S_scanner_to_persp", len(scanner_ids), len(scanner_ids),
                 len(train_samples), len(gallery), len(probe))
    return train_samples, gallery, probe, len(scanner_ids)


def parse_setting_paired_conditions(cond_a, cond_b, cond_paths, scanner_paths, seed):
    paths_a = cond_paths.get(cond_a, {}); paths_b = cond_paths.get(cond_b, {})
    if not paths_a: raise ValueError(f"No images for condition '{cond_a}'")
    if not paths_b: raise ValueError(f"No images for condition '{cond_b}'")
    eligible_ids = sorted(set(paths_a.keys()) & set(paths_b.keys()))
    if not eligible_ids: raise ValueError(f"No IDs with both '{cond_a}' and '{cond_b}'")
    label_map     = {ident: i for i, ident in enumerate(eligible_ids)}
    train_samples = []
    for cond, cond_dict in cond_paths.items():
        if cond in (cond_a, cond_b): continue
        for ident in eligible_ids:
            for p in cond_dict.get(ident, []):
                train_samples.append((p, label_map[ident]))
    for ident in eligible_ids:
        for p in scanner_paths.get(ident, []):
            train_samples.append((p, label_map[ident]))
    gallery = _all_samples({i: paths_a[i] for i in eligible_ids}, label_map)
    probe   = _all_samples({i: paths_b[i] for i in eligible_ids}, label_map)
    _print_stats(f"S_{cond_a}_{cond_b}", len(eligible_ids), len(eligible_ids),
                 len(train_samples), len(gallery), len(probe))
    return train_samples, gallery, probe, len(eligible_ids)


def _print_stats(name, n_train_ids, n_test_ids, train_n, gallery_n, probe_n):
    log(f"  [{name}]")
    log(f"    Train IDs / Test IDs  : {n_train_ids} / {n_test_ids}")
    log(f"    Train images          : {train_n}")
    log(f"    Gallery / Probe       : {gallery_n} / {probe_n}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (unchanged from CASIA-MS version)
# ─────────────────────────────────────────────────────────────────────────────

class DINOFinetune(nn.Module):
    """
    DINOv2 ViT-S/14 (facebookresearch/dinov2):
      - patch embed + blocks 0-9 : frozen
      - blocks 10-11             : trainable  (last 2 of 12 blocks)
    Returns L2-normalised [CLS] token embeddings (dim=384).
    """
    def __init__(self):
        super().__init__()
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14")

        # Freeze all, then selectively unfreeze blocks 10 and 11
        for name, p in backbone.named_parameters():
            p.requires_grad = False
            if "blocks.10" in name or "blocks.11" in name:
                p.requires_grad = True

        self.backbone  = backbone
        self.embed_dim = 384   # ViT-S/14 embedding dimension

    def forward(self, x):
        out = self.backbone.forward_features(x)
        cls = out["x_norm_clstoken"]    # [B, 384]  CLS token
        return F.normalize(cls, p=2, dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out))
    def forward(self, x): return F.normalize(self.head(x), dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    feats, labels = [], []
    for imgs, lbl in loader:
        feats.append(model(imgs.to(DEVICE)).cpu().numpy())
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


def evaluate(model, gallery_loader, probe_loader, out_dir=".", tag="eval"):
    gal_feats, gal_labels = extract_embeddings(model, gallery_loader)
    prb_feats, prb_labels = extract_embeddings(model, probe_loader)

    sim    = prb_feats @ gal_feats.T
    rank1  = 100.0 * (gal_labels[sim.argmax(axis=1)] == prb_labels).mean()

    scores_list, labels_list = [], []
    for i in range(len(prb_labels)):
        for j in range(len(gal_labels)):
            scores_list.append(float(sim[i, j]))
            labels_list.append(1 if prb_labels[i] == gal_labels[j] else -1)

    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _     = compute_eer_metric(scores_arr)

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

    train_loader = make_train_loader(train_samples)
    gal_loader   = make_eval_loader(gallery_samples)
    prb_loader   = make_eval_loader(probe_samples)

    model = DINOFinetune().to(DEVICE)
    proj  = ProjectionHead(model.embed_dim).to(DEVICE)

    criterion_arc    = pml_losses.ArcFaceLoss(
        num_classes=num_classes, embedding_size=model.embed_dim,
        margin=MARGIN, scale=SCALE).to(DEVICE)
    criterion_supcon = pml_losses.SupConLoss(temperature=0.1).to(DEVICE)

    all_params = (list(model.parameters()) +
                  list(proj.parameters()) +
                  list(criterion_arc.parameters()))

    optimizer = optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Classes={num_classes}  Train={len(train_samples)}  "
        f"Gallery={len(gallery_samples)}  Probe={len(probe_samples)}")
    log(f"  Trainable params: {trainable/1e6:.2f}M")

    best_rank1 = 0.0
    ckpt_path  = os.path.join(results_dir, "best_model.pth")

    # Pre-training baseline
    evaluate(model, gal_loader, prb_loader, rst_eval, "ep000_pretrain")

    for epoch in range(1, EPOCHS + 1):
        model.train(); proj.train(); criterion_arc.train()
        ep_loss = 0.0; ep_arc = 0.0; ep_con = 0.0
        ep_corr = 0;   ep_tot = 0

        for img_orig, aug1, aug2, aug3, y_i in train_loader:
            img_orig = img_orig.to(DEVICE)
            aug1     = aug1.to(DEVICE)
            aug2     = aug2.to(DEVICE)
            aug3     = aug3.to(DEVICE)
            y_i      = y_i.to(DEVICE)

            # Stack: original + 3 augmented views → 4× batch
            # original: stable anchor (no augmentation)
            # aug1-3  : three independently sampled CompNet-style augmentations
            imgs_all = torch.cat([img_orig, aug1, aug2, aug3], dim=0)
            y_all    = torch.cat([y_i, y_i, y_i, y_i], dim=0)

            optimizer.zero_grad()
            emb_all  = model(imgs_all)        # [2B, D]  L2-normalised
            proj_all = proj(emb_all)          # [2B, 128]

            loss_arc = criterion_arc(emb_all, y_all)
            loss_con = criterion_supcon(proj_all, y_all)
            loss     = loss_arc + LAMB * loss_con

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()

            ep_loss += loss.item(); ep_arc += loss_arc.item()
            ep_con  += loss_con.item()

            with torch.no_grad():
                preds    = criterion_arc.get_logits(emb_all).argmax(dim=1)
                ep_corr += (preds == y_all).sum().item()
                ep_tot  += y_all.size(0)

        scheduler.step()

        n       = len(train_loader)
        avg_acc = 100.0 * ep_corr / ep_tot
        log(f"  ep {epoch:03d}/{EPOCHS}  "
            f"loss={ep_loss/n:.4f}  arc={ep_arc/n:.4f}  "
            f"con={ep_con/n:.4f}  acc={avg_acc:.2f}%")

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            cur_eer, cur_rank1 = evaluate(
                model, gal_loader, prb_loader,
                rst_eval, f"ep{epoch:04d}")
            if cur_rank1 > best_rank1:
                best_rank1 = cur_rank1
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "rank1": cur_rank1, "eer": cur_eer}, ckpt_path)
                log(f"  *** New best Rank-1: {best_rank1:.2f}% ***")

    # Reload best and report final result
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_eer, final_rank1 = evaluate(
        model, gal_loader, prb_loader, rst_eval, "FINAL")
    log(f"  Best Rank-1={best_rank1:.2f}%  "
        f"Final: EER={final_eer*100:.4f}%  Rank-1={final_rank1:.2f}%")
    return final_eer, final_rank1


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_summary(all_results, out_path):
    col_w  = 14
    header = (f"{'Setting':<22}{'Train domain':<38}{'Test domain':<26}"
              f"{'EER (%)':>{col_w}}{'Rank-1 (%)':>{col_w}}")
    sep = "─" * len(header)
    lines = ["\nCross-Domain Closed-Set Results — Palm-Auth (DINOv2)",
             sep, header, sep]
    for r in all_results:
        eer_str   = f"{r['eer']*100:.2f}" if r['eer']   is not None else "—"
        rank1_str = f"{r['rank1']:.2f}"   if r['rank1'] is not None else "—"
        lines.append(f"{r['setting']:<22}{r['train_desc']:<38}{r['test_desc']:<26}"
                     f"{eer_str:>{col_w}}{rank1_str:>{col_w}}")
    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f: f.write(text + "\n")
    log(f"Summary saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    log("=" * 72)
    log(f"DINOv2 ViT-S/14 — Cross-Domain Closed-Set (Palm-Auth)")
    log(f"Device    : {DEVICE}")
    log(f"Epochs    : {EPOCHS}  LR: {LR}  λ_SupCon: {LAMB}")
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
            train_s, gal_s, prb_s, n_cls = s["parser"]()
            eer, rank1 = run_experiment(
                train_s, gal_s, prb_s, n_cls, results_dir)
            elapsed = time.time() - t_start
            log(f"\n  ✓  {s['label']}:  EER={eer*100:.4f}%  "
                f"Rank-1={rank1:.2f}%  Time={elapsed/60:.1f} min")
            with open(os.path.join(results_dir, "results.json"), "w") as f:
                json.dump({"setting": s["label"], "train_desc": s["train_desc"],
                           "test_desc": s["test_desc"], "num_classes": n_cls,
                           "EER_pct": eer*100, "Rank1_pct": rank1}, f, indent=2)
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"],
                                 "eer": eer, "rank1": rank1})
        except Exception as e:
            import traceback; traceback.print_exc()
            log(f"\n  ✗  {s['label']} FAILED: {e}")
            all_results.append({"setting": s["label"], "train_desc": s["train_desc"],
                                 "test_desc": s["test_desc"],
                                 "eer": None, "rank1": None})

    log(f"\n\n{'='*72}")
    log(f"ALL {len(SETTINGS)} SETTINGS COMPLETE")
    log(f"{'='*72}")
    print_and_save_summary(
        all_results, os.path.join(SAVE_DIR, "results_summary.txt"))


if __name__ == "__main__":
    main()
