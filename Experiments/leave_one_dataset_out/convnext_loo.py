"""
ConvNeXtV2-Tiny — Leave-One-Out Cross-Dataset Experiment Runner
================================================================
Trains on ALL combinations of three datasets and evaluates on the
left-out fourth dataset. Four experiments total:

  Train                              Test
  ─────────────────────────────────  ──────────
  CASIA-MS + MPDv2   + XJTU          X-Palm
  X-Palm + MPDv2  + XJTU          CASIA-MS
  X-Palm + CASIA-MS + XJTU        MPDv2
  X-Palm + CASIA-MS + MPDv2       XJTU

Results are saved to:
  {BASE_RESULTS_DIR}/test_{Y}/         ← per-experiment outputs
  {BASE_RESULTS_DIR}/results_table.txt ← final EER / Rank-1 table
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
    "casiams_data_root"    : "./CASIA-MS-ROI",
    "x_palm_data_root"  : "./x_palm_roi",
    "mpd_data_root"        : "./MPDv2_roi",
    "xjtu_data_root"       : "./XJTU-UP",

    "test_gallery_ratio"   : 0.50,
    "use_scanner"          : True,

    "img_side"             : 112,
    "batch_size"           : 32,
    "num_epochs"           : 200,
    "lr"                   : 1e-3,
    "weight_decay"         : 1e-4,
    "lamb"                 : 0.2,
    "margin"               : 0.3,
    "scale"                : 16,
    "eval_every"           : 5,
    "num_workers"          : 4,

    "base_results_dir"     : "./rst_convnext_loo",
    "random_seed"          : 42,
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
import torch.optim as optim
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses as pml_losses

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

ALLOWED_SPECTRA = {"green", "ir", "yellow", "pink", "white"}

N_HIGH = 160; N_LOW = 40
TARGET_HIGH_CASIA = 33; TARGET_LOW_CASIA = 15
TARGET_HIGH_XJTU  = 33; TARGET_LOW_XJTU  = 15

XJTU_VARIATIONS = [
    ("iPhone", "Flash"), ("iPhone", "Nature"),
    ("huawei", "Flash"), ("huawei", "Nature"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
#  MODEL  (unchanged)
# ══════════════════════════════════════════════════════════════

class ConvNeXtFinetune(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = timm.create_model('convnextv2_tiny', pretrained=True, num_classes=0)
        for p in backbone.parameters(): p.requires_grad = False
        for p in backbone.stages[3].parameters(): p.requires_grad = True
        if hasattr(backbone, 'norm'):
            for p in backbone.norm.parameters(): p.requires_grad = True
        self.backbone  = backbone
        self.embed_dim = backbone.num_features

    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_out))
    def forward(self, x): return F.normalize(self.head(x), dim=1)


# ══════════════════════════════════════════════════════════════
#  AUGMENTATION & DATASETS  (unchanged)
# ══════════════════════════════════════════════════════════════

def _base_transform(img_side):
    return transforms.Compose([
        transforms.Resize(img_side),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def _make_aug(img_side):
    return transforms.Compose([
        transforms.Resize(img_side),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            transforms.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.0,1.0)),
            transforms.RandomPerspective(distortion_scale=0.15, p=1),
            transforms.RandomChoice([
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(int(0.5*img_side), 0)),
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(0, int(0.5*img_side))),
            ]),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TrainDataset(Dataset):
    def __init__(self, samples, img_side=112):
        self.samples  = samples; self.img_side = img_side
        self.base     = _base_transform(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return (self.base(img),
                _make_aug(self.img_side)(img),
                _make_aug(self.img_side)(img),
                _make_aug(self.img_side)(img),
                label)


class EvalDataset(Dataset):
    def __init__(self, samples, img_side=112):
        self.samples   = samples
        self.transform = _base_transform(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def make_train_loader(samples, batch_size, img_side, num_workers):
    return DataLoader(TrainDataset(samples, img_side),
                      batch_size=min(batch_size, len(samples)),
                      shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=len(samples) > batch_size)

def make_eval_loader(samples, img_side, num_workers):
    return DataLoader(EvalDataset(samples, img_side),
                      batch_size=min(128, len(samples)),
                      shuffle=False, num_workers=num_workers, pin_memory=True)


# ══════════════════════════════════════════════════════════════
#  DATASET PARSERS  (from CompNet LOO)
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root, seed=42):
    rng     = random.Random(seed)
    id_spec = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) < 4: continue
        id_spec[parts[0]+"_"+parts[1]][parts[2]].append(os.path.join(data_root, fname))
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
    print(f"  [CASIA-MS] ids={len(id2paths)}  total={sum(len(v) for v in id2paths.values())}")
    return id2paths


def parse_palm_auth_data(data_root, use_scanner=False, seed=42):
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
    all_ids = sorted(id2paths.keys(), key=lambda i: len(id2paths[i]), reverse=True)
    if len(all_ids) < N_HIGH + N_LOW:
        raise ValueError(f"Palm-Auth: need {N_HIGH+N_LOW} IDs, found {len(all_ids)}")
    selected_ids = all_ids[:N_HIGH + N_LOW]
    result = {k: list(id2paths[k]) for k in selected_ids}
    counts = [len(v) for v in result.values()]
    print(f"  [Palm-Auth] ids={len(result)}  total={sum(counts)}  cutoff={counts[-1]}")
    return result


def parse_mpd_data(data_root, seed=42):
    id_dev = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        parts = os.path.splitext(fname)[0].split("_")
        if len(parts) != 5: continue
        subject, session, device, hand_side, iteration = parts
        if device not in ("h","m") or hand_side not in ("l","r"): continue
        id_dev[subject+"_"+hand_side][device].append(os.path.join(data_root, fname))
    all_ids = sorted(id_dev.keys(),
                     key=lambda i: len(id_dev[i].get("h",[]))+len(id_dev[i].get("m",[])),
                     reverse=True)
    if len(all_ids) < N_HIGH + N_LOW:
        raise ValueError(f"MPDv2: need {N_HIGH+N_LOW} IDs, found {len(all_ids)}")
    selected_ids = all_ids[:N_HIGH + N_LOW]
    id2paths = {ident: id_dev[ident].get("h",[]) + id_dev[ident].get("m",[])
                for ident in selected_ids}
    counts = [len(v) for v in id2paths.values()]
    print(f"  [MPDv2] ids={len(id2paths)}  total={sum(counts)}  cutoff={counts[-1]}")
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
                id_var[id_folder][(device, condition)].append(os.path.join(id_dir, fname))
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
    print(f"  [XJTU] ids={len(id2paths)}  total={sum(len(v) for v in id2paths.values())}")
    return id2paths


def get_parser(dataset_name, cfg):
    name = _ds_key(dataset_name); seed = cfg["random_seed"]
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
#  COMBINED TRAINING SET BUILDER  (from CompNet LOO)
# ══════════════════════════════════════════════════════════════

def build_combined_train_samples(train_datasets, cfg):
    train_samples = []; label_offset = 0
    for ds_name in train_datasets:
        print(f"  Parsing {ds_name} (train) …")
        id2paths   = get_parser(ds_name, cfg)()
        sorted_ids = sorted(id2paths.keys())
        label_map  = {ident: label_offset + i for i, ident in enumerate(sorted_ids)}
        for ident in sorted_ids:
            for path in id2paths[ident]:
                train_samples.append((path, label_map[ident]))
        n_subj = len(sorted_ids); n_imgs = sum(len(id2paths[i]) for i in sorted_ids)
        print(f"    → {n_subj} subjects | {n_imgs} images "
              f"| labels {label_offset}–{label_offset + n_subj - 1}")
        label_offset += n_subj
    print(f"  Combined train: {label_offset} total subjects | "
          f"{len(train_samples)} total images")
    return train_samples, label_offset


# ══════════════════════════════════════════════════════════════
#  TEST SPLIT  (from CompNet LOO)
# ══════════════════════════════════════════════════════════════

def split_test_dataset(id2paths, gallery_ratio=0.50, seed=42):
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
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(model, proj, criterion_arc, num_classes,
                               cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    weights_path = os.path.join(cache_dir,
                                f"init_weights_ConvNeXt_nc{num_classes}.pth")
    if os.path.exists(weights_path):
        print(f"  Loading cached init weights: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        proj.load_state_dict(ckpt["proj"])
        criterion_arc.load_state_dict(ckpt["arc"])
    else:
        print(f"  Saving init weights: {weights_path}")
        torch.save({"model": model.state_dict(),
                    "proj":  proj.state_dict(),
                    "arc":   criterion_arc.state_dict()}, weights_path)


# ══════════════════════════════════════════════════════════════
#  EVALUATION  (unchanged)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval(); feats, labels = [], []
    for imgs, lbl in loader:
        feats.append(model(imgs.to(DEVICE)).cpu().numpy())
        labels.append(lbl.numpy())
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


def evaluate(model, gallery_loader, probe_loader, out_dir=".", tag="eval"):
    gal_feats, gal_labels = extract_embeddings(model, gallery_loader)
    prb_feats, prb_labels = extract_embeddings(model, probe_loader)
    sim   = prb_feats @ gal_feats.T
    rank1 = 100.0 * (gal_labels[sim.argmax(axis=1)] == prb_labels).mean()
    scores_list, labels_list = [], []
    for i in range(len(prb_labels)):
        for j in range(len(gal_labels)):
            scores_list.append(float(sim[i, j]))
            labels_list.append(1 if prb_labels[i] == gal_labels[j] else -1)
    scores_arr = np.column_stack([scores_list, labels_list])
    eer, _     = compute_eer(scores_arr)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"scores_{tag}.txt"), "w") as f:
        for s, l in zip(scores_list, labels_list): f.write(f"{s} {l}\n")
    print(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer, rank1


# ══════════════════════════════════════════════════════════════
#  SINGLE EXPERIMENT
# ══════════════════════════════════════════════════════════════

def run_experiment(train_datasets, test_dataset, cfg, device=None):
    seed           = cfg["random_seed"]
    results_dir    = cfg["results_dir"]
    img_side       = cfg["img_side"]
    batch_size     = cfg["batch_size"]
    num_epochs     = cfg["num_epochs"]
    lr             = cfg["lr"]
    weight_decay   = cfg["weight_decay"]
    lamb           = cfg["lamb"]
    margin         = cfg["margin"]
    scale          = cfg["scale"]
    test_gal_ratio = cfg["test_gallery_ratio"]
    eval_every     = cfg["eval_every"]
    nw             = cfg["num_workers"]
    cache_dir      = cfg["base_results_dir"]
    eval_tag_base  = test_dataset.replace("-", "")

    os.makedirs(results_dir, exist_ok=True)
    rst_eval = os.path.join(results_dir, "eval")
    os.makedirs(rst_eval, exist_ok=True)

    train_samples, num_classes = build_combined_train_samples(train_datasets, cfg)

    print(f"  Parsing {test_dataset} (test) …")
    test_id2paths  = get_parser(test_dataset, cfg)()
    gallery_samples, probe_samples = split_test_dataset(
        test_id2paths, test_gal_ratio, seed)

    train_loader   = make_train_loader(train_samples, batch_size, img_side, nw)
    gallery_loader = make_eval_loader(gallery_samples, img_side, nw)
    probe_loader   = make_eval_loader(probe_samples,   img_side, nw)

    print(f"  Gallery={len(gallery_samples)}  Probe={len(probe_samples)}  "
          f"Classes(train)={num_classes}")

    model = ConvNeXtFinetune().to(device)
    proj  = ProjectionHead(model.embed_dim).to(device)

    criterion_arc    = pml_losses.ArcFaceLoss(
        num_classes=num_classes, embedding_size=model.embed_dim,
        margin=margin, scale=scale).to(device)
    criterion_supcon = pml_losses.SupConLoss(temperature=0.1).to(device)

    get_or_create_init_weights(model, proj, criterion_arc,
                               num_classes, cache_dir, device)

    all_params = (list(model.parameters()) +
                  list(proj.parameters()) +
                  list(criterion_arc.parameters()))
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable/1e6:.2f}M")

    best_rank1 = 0.0
    ckpt_path  = os.path.join(results_dir, "best_model.pth")

    evaluate(model, gallery_loader, probe_loader,
             rst_eval, f"ep000_pretrain_{eval_tag_base}")

    for epoch in range(1, num_epochs + 1):
        model.train(); proj.train(); criterion_arc.train()
        ep_loss = 0.0; ep_arc = 0.0; ep_con = 0.0
        ep_corr = 0;   ep_tot = 0

        for img_orig, aug1, aug2, aug3, y_i in train_loader:
            img_orig = img_orig.to(device); aug1 = aug1.to(device)
            aug2     = aug2.to(device);     aug3 = aug3.to(device)
            y_i      = y_i.to(device)
            imgs_all = torch.cat([img_orig, aug1, aug2, aug3], dim=0)
            y_all    = torch.cat([y_i, y_i, y_i, y_i], dim=0)

            optimizer.zero_grad()
            emb_all  = model(imgs_all); proj_all = proj(emb_all)
            loss_arc = criterion_arc(emb_all, y_all)
            loss_con = criterion_supcon(proj_all, y_all)
            loss     = loss_arc + lamb * loss_con
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()

            ep_loss += loss.item(); ep_arc += loss_arc.item(); ep_con += loss_con.item()
            with torch.no_grad():
                preds    = criterion_arc.get_logits(emb_all).argmax(dim=1)
                ep_corr += (preds == y_all).sum().item(); ep_tot += y_all.size(0)

        scheduler.step()
        n = len(train_loader)
        print(f"  ep {epoch:03d}/{num_epochs}  "
              f"loss={ep_loss/n:.4f}  arc={ep_arc/n:.4f}  "
              f"con={ep_con/n:.4f}  acc={100.*ep_corr/ep_tot:.2f}%")

        if epoch % eval_every == 0 or epoch == num_epochs:
            cur_eer, cur_rank1 = evaluate(
                model, gallery_loader, probe_loader,
                rst_eval, f"ep{epoch:04d}_{eval_tag_base}")
            if cur_rank1 > best_rank1:
                best_rank1 = cur_rank1
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "rank1": cur_rank1, "eer": cur_eer}, ckpt_path)
                print(f"  *** New best Rank-1: {best_rank1:.2f}% ***")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_eer, final_rank1 = evaluate(
        model, gallery_loader, probe_loader,
        rst_eval, f"FINAL_{eval_tag_base}")
    return final_eer, final_rank1


# ══════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ══════════════════════════════════════════════════════════════

def print_and_save_table(results, all_datasets, out_path):
    col_w  = 16
    header = (f"{'Test (left out)':<18}"
              f"{'Train datasets':<44}"
              f"{'EER (%)':>{col_w}}"
              f"{'Rank-1 (%)':>{col_w}}")
    sep    = "─" * len(header)
    lines  = ["\nLeave-One-Out Results — ConvNeXtV2-Tiny", sep, header, sep]
    eer_vals, rank1_vals = [], []
    for test_ds in all_datasets:
        train_ds  = [d for d in all_datasets if d != test_ds]
        train_str = " + ".join(d.replace("-","") for d in train_ds)
        val = results.get(test_ds)
        if val is not None:
            eer_str = f"{val[0]:.2f}"; rank1_str = f"{val[1]:.2f}"
            eer_vals.append(val[0]);    rank1_vals.append(val[1])
        else:
            eer_str = rank1_str = "—"
        lines.append(f"{test_ds.replace('-',''):<18}"
                     f"{train_str:<44}"
                     f"{eer_str:>{col_w}}"
                     f"{rank1_str:>{col_w}}")
    lines.append(sep)
    avg_eer   = f"{sum(eer_vals)/len(eer_vals):.2f}"     if eer_vals   else "—"
    avg_rank1 = f"{sum(rank1_vals)/len(rank1_vals):.2f}" if rank1_vals else "—"
    lines.append(f"{'Avg':<18}{'':44}{avg_eer:>{col_w}}{avg_rank1:>{col_w}}")
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
    print(f"  ConvNeXtV2-Tiny — Leave-One-Out Cross-Dataset Experiment")
    print(f"  Device      : {device}")
    print(f"  Datasets    : {ALL_DATASETS}")
    print(f"  Strategy    : train on 3, test on left-out 1")
    print(f"  Epochs      : {BASE_CONFIG['num_epochs']}")
    print(f"  Results dir : {base_results_dir}")
    print(f"{'='*60}\n")

    n_total  = len(ALL_DATASETS)
    n_done   = 0; results = {}; failures = []

    for test_dataset in ALL_DATASETS:
        train_datasets = [d for d in ALL_DATASETS if d != test_dataset]
        n_done += 1
        train_str = " + ".join(train_datasets)
        exp_label = f"train=[{train_str}]  test={test_dataset}"

        print(f"\n{'='*60}")
        print(f"  Experiment {n_done}/{n_total}:  {exp_label}")
        print(f"{'='*60}")

        cfg = copy.deepcopy(BASE_CONFIG)
        safe_test          = test_dataset.replace("-","").replace(" ","")
        cfg["results_dir"] = os.path.join(base_results_dir, f"test_{safe_test}")

        t_start = time.time()
        try:
            eer, rank1 = run_experiment(
                train_datasets, test_dataset, cfg, device=device)
            results[test_dataset] = (eer * 100, rank1)
            elapsed = time.time() - t_start
            print(f"\n  ✓  {exp_label}")
            print(f"     EER={eer*100:.4f}%  Rank-1={rank1:.2f}%  "
                  f"Time={elapsed/60:.1f} min")
        except Exception as e:
            results[test_dataset] = None
            failures.append((test_dataset, str(e)))
            print(f"\n  ✗  {exp_label}  FAILED: {e}")

    table_path = os.path.join(base_results_dir, "results_table.txt")
    print(f"\n\n{'='*60}"); print(f"  ALL EXPERIMENTS COMPLETE"); print(f"{'='*60}")
    print_and_save_table(results, ALL_DATASETS, table_path)

    if failures:
        print(f"\nFailed experiments ({len(failures)}):")
        for te, err in failures: print(f"  test={te}  → {err}")

    json_results = {te: list(v) if v else None for te, v in results.items()}
    with open(os.path.join(base_results_dir, "results_raw.json"), "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nRaw results saved to: "
          f"{os.path.join(base_results_dir, 'results_raw.json')}")


if __name__ == "__main__":
    main()
