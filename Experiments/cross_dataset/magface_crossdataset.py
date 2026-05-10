"""
MagFace iResNet100 — Full Cross-Dataset Experiment Runner
==========================================================
Backbone : iResNet100 pretrained on MS1MV2 (MagFace checkpoint)
           first 75% of tensors frozen
Loss     : MagFace adaptive margin + magnitude regularization
Input    : 112×112 RGB, mean=0.5, std=0.5

Train datasets : CASIA-MS | X-Palm | MPDv2 | XJTU
Test  datasets : CASIA-MS | X-Palm | MPDv2 | XJTU

Results saved to:
  {BASE_RESULTS_DIR}/train_{X}_test_{Y}/
  {BASE_RESULTS_DIR}/results_table.txt
  {BASE_RESULTS_DIR}/results_raw.json
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
    "pretrained_weights"   : "./checkpoints/magface_iresnet100.pth",

    "train_subject_ratio"  : 0.80,
    "test_gallery_ratio"   : 0.50,
    "use_scanner"          : True,

    # MagFace loss parameters (original paper)
    "arcface_s"            : 64.0,
    "m_l"                  : 0.45,
    "m_u"                  : 0.80,
    "l_a"                  : 10.0,
    "u_a"                  : 110.0,
    "lambda_g"             : 20.0,

    "img_side"             : 112,
    "batch_size"           : 32,
    "num_epochs"           : 200,
    "lr"                   : 1e-4,
    "weight_decay"         : 5e-4,
    "eval_every"           : 5,
    "num_workers"          : 4,

    "base_results_dir"     : "./rst_magface_crossdataset",
    "random_seed"          : 42,
}
# ==============================================================

import os, copy, json, math, time, random, warnings
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

ALLOWED_SPECTRA = {"green", "ir", "yellow", "pink", "white"}
N_HIGH = 160; N_LOW = 40
TARGET_HIGH_CASIA = 33; TARGET_LOW_CASIA = 15
TARGET_HIGH_MPD   = 33; TARGET_LOW_MPD   = 15
TARGET_HIGH_XJTU  = 33; TARGET_LOW_XJTU  = 15
XJTU_VARIATIONS = [
    ("iPhone", "Flash"), ("iPhone", "Nature"),
    ("huawei", "Flash"), ("huawei", "Nature"),
]
IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ══════════════════════════════════════════════════════════════
#  iResNet100 BACKBONE  (unchanged)
# ══════════════════════════════════════════════════════════════

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2   = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3   = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample; self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x);    out = self.conv1(out)
        out = self.bn2(out);  out = self.prelu(out)
        out = self.conv2(out); out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    def __init__(self, block, layers, dropout=0.0, num_features=512,
                 groups=1, width_per_group=64):
        super().__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu    = nn.PReLU(64)
        self.layer1   = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2   = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3   = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4   = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2      = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout  = nn.Dropout(p=dropout)
        self.fc       = nn.Linear(512*7*7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                nn.BatchNorm2d(planes*block.expansion, eps=1e-05))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.bn2(x);    x = self.dropout(x)
        x = x.flatten(1);   x = self.fc(x)
        x = self.features(x)
        return x


def iresnet100(num_features=512):
    return IResNet(IBasicBlock, [3, 13, 30, 3], num_features=num_features)


class MagFaceBackbone(nn.Module):
    """
    iResNet100 loaded from MagFace checkpoint.
    Freeze ratio: first 75% of parameter tensors.
    Returns RAW (non-normalised) embeddings — MagFace needs the magnitude.
    """
    def __init__(self, pretrained_path, freeze_ratio=0.75):
        super().__init__()
        self.net = iresnet100()
        if pretrained_path and os.path.exists(pretrained_path):
            ckpt  = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("features.module.", ""): v
                     for k, v in state.items()
                     if k.startswith("features.module.")}
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            print(f"  Loaded: {pretrained_path}  "
                  f"(epoch {ckpt.get('epoch','?')}  arch={ckpt.get('arch','?')})")
            if missing:    print(f"    Missing keys    : {len(missing)}")
            if unexpected: print(f"    Unexpected keys : {len(unexpected)}")
        else:
            print(f"  [WARN] Pretrained weights not found: {pretrained_path}")

        all_params = list(self.net.parameters())
        n_freeze   = int(len(all_params) * freeze_ratio)
        for i, p in enumerate(all_params):
            p.requires_grad = (i >= n_freeze)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    def forward(self, x):
        return self.net(x)   # raw, not normalised


# ══════════════════════════════════════════════════════════════
#  MAGFACE LOSS  (unchanged)
# ══════════════════════════════════════════════════════════════

class MagFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=512,
                 s=64.0, m_l=0.45, m_u=0.80,
                 l_a=10.0, u_a=110.0, lambda_g=20.0):
        super().__init__()
        self.s=s; self.m_l=m_l; self.m_u=m_u
        self.l_a=l_a; self.u_a=u_a; self.lambda_g=lambda_g
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def _adaptive_margin(self, norm):
        a = norm.clamp(self.l_a, self.u_a)
        return self.m_l + (self.m_u-self.m_l)*(a-self.l_a)/(self.u_a-self.l_a)

    def _magnitude_regularizer(self, norm):
        a = norm.clamp(self.l_a, self.u_a)
        return ((1.0/(self.u_a**2))*a + 1.0/a).mean()

    def forward(self, embeddings, labels):
        norm      = embeddings.norm(dim=1)
        z_normed  = F.normalize(embeddings, p=2, dim=1)
        W_normed  = F.normalize(self.weight, p=2, dim=1)
        cos_theta = (z_normed @ W_normed.t()).clamp(-1+1e-7, 1-1e-7)
        m         = self._adaptive_margin(norm)
        sin_theta = (1.0-cos_theta**2).sqrt()
        cos_m=torch.cos(m).unsqueeze(1); sin_m=torch.sin(m).unsqueeze(1)
        th=math.cos(math.pi-self.m_u); mm=math.sin(math.pi-self.m_u)*self.m_u
        cos_theta_m = cos_theta*cos_m - sin_theta*sin_m
        cos_theta_m = torch.where(cos_theta>th, cos_theta_m, cos_theta-mm)
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1,1), 1.0)
        logits  = self.s*(one_hot*cos_theta_m + (1-one_hot)*cos_theta)
        L_arc   = self.ce(logits, labels)
        L_g     = self._magnitude_regularizer(norm)
        return L_arc + self.lambda_g*L_g, L_arc.item(), L_g.item()

    @torch.no_grad()
    def get_logits(self, embeddings):
        z=F.normalize(embeddings, p=2, dim=1)
        W=F.normalize(self.weight, p=2, dim=1)
        return self.s*(z@W.t())


# ══════════════════════════════════════════════════════════════
#  TRANSFORMS & DATASETS
# ══════════════════════════════════════════════════════════════

def _base_tf(img_side):
    return transforms.Compose([
        transforms.Resize((img_side, img_side)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

def _aug_tf(img_side):
    return transforms.Compose([
        transforms.Resize((img_side, img_side)),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
            transforms.RandomResizedCrop(img_side, scale=(0.8,1.0), ratio=(1.0,1.0)),
            transforms.RandomPerspective(distortion_scale=0.15, p=1),
            transforms.RandomChoice([
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(int(0.5*img_side),0)),
                transforms.RandomRotation(10, interpolation=Image.BICUBIC,
                                          expand=False, center=(0,int(0.5*img_side))),
            ]),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])


class TrainDataset(Dataset):
    def __init__(self, samples, img_side):
        self.samples=samples; self.transform=_aug_tf(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


class EvalDataset(Dataset):
    def __init__(self, samples, img_side):
        self.samples=samples; self.transform=_base_tf(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def make_train_loader(samples, cfg):
    return DataLoader(TrainDataset(samples, cfg["img_side"]),
                      batch_size=min(cfg["batch_size"], len(samples)),
                      shuffle=True, num_workers=cfg["num_workers"],
                      pin_memory=True, drop_last=len(samples)>cfg["batch_size"])

def make_eval_loader(samples, cfg):
    return DataLoader(EvalDataset(samples, cfg["img_side"]),
                      batch_size=min(128, len(samples)),
                      shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)


# ══════════════════════════════════════════════════════════════
#  DATASET PARSERS
# ══════════════════════════════════════════════════════════════

def parse_casia_ms(data_root, seed=42):
    rng=random.Random(seed); id_spec=defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        parts=os.path.splitext(fname)[0].split("_")
        if len(parts)<4: continue
        id_spec[parts[0]+"_"+parts[1]][parts[2]].append(os.path.join(data_root,fname))
    all_ids=sorted(id_spec.keys())
    if len(all_ids)<N_HIGH+N_LOW: raise ValueError(f"CASIA-MS: need {N_HIGH+N_LOW}")
    selected=sorted(rng.sample(all_ids,N_HIGH+N_LOW)); rng.shuffle(selected)
    high_ids=selected[:N_HIGH]; low_ids=selected[N_HIGH:]
    def _sample(ident,target):
        spec_list=list(sorted(id_spec[ident].keys())); rng.shuffle(spec_list)
        n_spec=len(spec_list); base_s=target//n_spec; rem_s=target%n_spec
        chosen=[]
        for j,sp in enumerate(spec_list):
            k=min(base_s+(1 if j<rem_s else 0),len(id_spec[ident][sp]))
            chosen.extend(rng.sample(id_spec[ident][sp],k))
        return chosen
    id2paths={}
    for ident in high_ids: id2paths[ident]=_sample(ident,TARGET_HIGH_CASIA)
    for ident in low_ids:  id2paths[ident]=_sample(ident,TARGET_LOW_CASIA)
    print(f"  [CASIA-MS] ids={len(id2paths)}  total={sum(len(v) for v in id2paths.values())}")
    return id2paths


def parse_palm_auth_data(data_root, use_scanner=False):
    id2paths=defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir=os.path.join(data_root,subject_id)
        if not os.path.isdir(subject_dir): continue
        roi_dir=os.path.join(subject_dir,"roi_perspective")
        if os.path.isdir(roi_dir):
            for fname in sorted(os.listdir(roi_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                parts=os.path.splitext(fname)[0].split("_")
                if len(parts)<3: continue
                id2paths[parts[0]+"_"+parts[1]].append(os.path.join(roi_dir,fname))
        if use_scanner:
            scan_dir=os.path.join(subject_dir,"roi_scanner")
            if os.path.isdir(scan_dir):
                for fname in sorted(os.listdir(scan_dir)):
                    if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                    parts=os.path.splitext(fname)[0].split("_")
                    if len(parts)<4: continue
                    if parts[2].lower() not in ALLOWED_SPECTRA: continue
                    id2paths[subject_id+"_"+parts[1].lower()].append(os.path.join(scan_dir,fname))
    result=dict(id2paths); counts=[len(v) for v in result.values()]
    print(f"  [Palm-Auth] ids={len(result)}  total={sum(counts)}")
    return result


def parse_mpd_data(data_root, seed=42):
    rng=random.Random(seed); id_dev=defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(data_root)):
        if not fname.lower().endswith((".jpg",".jpeg",".bmp",".png")): continue
        parts=os.path.splitext(fname)[0].split("_")
        if len(parts)!=5: continue
        subject,session,device,hand_side,iteration=parts
        if device not in ("h","m") or hand_side not in ("l","r"): continue
        id_dev[subject+"_"+hand_side][device].append(os.path.join(data_root,fname))
    all_ids=list(id_dev.keys()); rng.shuffle(all_ids)
    all_ids.sort(key=lambda i:len(id_dev[i].get("h",[]))+len(id_dev[i].get("m",[])),reverse=True)
    if len(all_ids)<N_HIGH: raise ValueError(f"MPDv2: need {N_HIGH} IDs")
    high_ids=all_ids[:N_HIGH]
    low_cands=[i for i in all_ids[N_HIGH:]
               if len(id_dev[i].get("h",[]))+len(id_dev[i].get("m",[]))>=TARGET_LOW_MPD]
    if len(low_cands)<N_LOW: raise ValueError("MPDv2: not enough low IDs")
    low_ids=low_cands[:N_LOW]
    def _sample(ident,target):
        paths=id_dev[ident].get("h",[])+id_dev[ident].get("m",[])
        return rng.sample(paths,min(target,len(paths)))
    id2paths={}
    for ident in high_ids: id2paths[ident]=_sample(ident,TARGET_HIGH_MPD)
    for ident in low_ids:  id2paths[ident]=_sample(ident,TARGET_LOW_MPD)
    print(f"  [MPDv2] ids={len(id2paths)}  total={sum(len(v) for v in id2paths.values())}")
    return id2paths


def parse_xjtu_data(data_root, seed=42):
    rng=random.Random(seed); id_var=defaultdict(lambda: defaultdict(list))
    for device,condition in XJTU_VARIATIONS:
        var_dir=os.path.join(data_root,device,condition)
        if not os.path.isdir(var_dir): print(f"  [XJTU] WARNING: {var_dir}"); continue
        for id_folder in sorted(os.listdir(var_dir)):
            id_dir=os.path.join(var_dir,id_folder)
            if not os.path.isdir(id_dir): continue
            parts=id_folder.split("_")
            if len(parts)<2 or parts[0].upper() not in ("L","R"): continue
            for fname in sorted(os.listdir(id_dir)):
                if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
                id_var[id_folder][(device,condition)].append(os.path.join(id_dir,fname))
    all_ids=sorted(id_var.keys())
    if len(all_ids)<N_HIGH+N_LOW: raise ValueError(f"XJTU: need {N_HIGH+N_LOW}")
    selected=sorted(rng.sample(all_ids,N_HIGH+N_LOW)); rng.shuffle(selected)
    high_ids=selected[:N_HIGH]; low_ids=selected[N_HIGH:]
    def _sample_var(ident,target):
        var_keys=list(XJTU_VARIATIONS); rng.shuffle(var_keys)
        n_var=len(var_keys); base_v=target//n_var; rem_v=target%n_var
        chosen=[]
        for j,vk in enumerate(var_keys):
            k=min(base_v+(1 if j<rem_v else 0),len(id_var[ident].get(vk,[])))
            if k>0: chosen.extend(rng.sample(id_var[ident].get(vk,[]),k))
        return chosen
    id2paths={}
    for ident in high_ids: id2paths[ident]=_sample_var(ident,TARGET_HIGH_XJTU)
    for ident in low_ids:  id2paths[ident]=_sample_var(ident,TARGET_LOW_XJTU)
    print(f"  [XJTU] ids={len(id2paths)}  total={sum(len(v) for v in id2paths.values())}")
    return id2paths


def get_parser(dataset_name, cfg):
    name=_ds_key(dataset_name); seed=cfg["random_seed"]
    if name=="casiams":    return lambda: parse_casia_ms(cfg["casiams_data_root"],seed=seed)
    elif name=="palmauth": return lambda: parse_palm_auth_data(cfg["palm_auth_data_root"],
                                                               use_scanner=cfg.get("use_scanner",False))
    elif name=="mpdv2":    return lambda: parse_mpd_data(cfg["mpd_data_root"],seed=seed)
    elif name=="xjtu":     return lambda: parse_xjtu_data(cfg["xjtu_data_root"],seed=seed)
    else: raise ValueError(f"Unknown dataset: '{dataset_name}'")

def _ds_key(name): return name.strip().lower().replace("-","").replace("_","")


# ══════════════════════════════════════════════════════════════
#  SPLITS
# ══════════════════════════════════════════════════════════════

def split_same_dataset(id2paths, train_subject_ratio=0.80, gallery_ratio=0.50, seed=42):
    rng=random.Random(seed); identities=sorted(id2paths.keys()); rng.shuffle(identities)
    n_train=max(1,int(len(identities)*train_subject_ratio))
    train_ids=identities[:n_train]; test_ids=identities[n_train:]
    train_label_map={k:i for i,k in enumerate(train_ids)}
    test_label_map ={k:i for i,k in enumerate(test_ids)}
    train_samples=[(p,train_label_map[ident]) for ident in train_ids for p in id2paths[ident]]
    gallery_samples,probe_samples=[],[]
    for ident in test_ids:
        paths=list(id2paths[ident]); rng.shuffle(paths)
        n_gal=max(1,int(len(paths)*gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p,test_label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p,test_label_map[ident]))
    return train_samples,gallery_samples,probe_samples,train_label_map,test_label_map


def split_cross_dataset_test(id2paths, gallery_ratio=0.50, seed=42):
    rng=random.Random(seed); label_map={k:i for i,k in enumerate(sorted(id2paths.keys()))}
    gallery_samples,probe_samples=[],[]
    for ident,paths in id2paths.items():
        paths=list(paths); rng.shuffle(paths)
        n_gal=max(1,int(len(paths)*gallery_ratio))
        for p in paths[:n_gal]: gallery_samples.append((p,label_map[ident]))
        for p in paths[n_gal:]: probe_samples.append((p,label_map[ident]))
    return gallery_samples,probe_samples,label_map


# ══════════════════════════════════════════════════════════════
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(model, criterion, num_classes, cache_dir, device):
    os.makedirs(cache_dir, exist_ok=True)
    weights_path=os.path.join(cache_dir,f"init_weights_MagFace_nc{num_classes}.pth")
    if os.path.exists(weights_path):
        print(f"  Loading cached init weights: {weights_path}")
        ckpt=torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        criterion.load_state_dict(ckpt["criterion"])
    else:
        print(f"  Saving init weights: {weights_path}")
        torch.save({"model":model.state_dict(),"criterion":criterion.state_dict()},weights_path)


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval(); feats,labels=[],[]
    for imgs,lbl in loader:
        raw=model(imgs.to(device))
        raw=torch.nan_to_num(raw,nan=0.0,posinf=0.0,neginf=0.0)
        emb=F.normalize(raw,p=2,dim=1)
        feats.append(emb.cpu().numpy()); labels.append(lbl.numpy())
    feats=np.concatenate(feats)
    if not np.isfinite(feats).all():
        feats=np.nan_to_num(feats,nan=0.0,posinf=0.0,neginf=0.0)
    return feats,np.concatenate(labels)


def compute_eer(scores_array):
    ins =scores_array[scores_array[:,1]==1,0]
    outs=scores_array[scores_array[:,1]==-1,0]
    if len(ins)==0 or len(outs)==0: return 1.0,0.0
    y=np.concatenate([np.ones(len(ins)),np.zeros(len(outs))])
    s=np.concatenate([ins,outs])
    if not np.isfinite(s).all() or np.unique(s).size<2: return 1.0,0.0
    fpr,tpr,thresholds=roc_curve(y,s,pos_label=1)
    eer=brentq(lambda x:1.0-x-interp1d(fpr,tpr)(x),0.0,1.0)
    thresh=float(interp1d(fpr,thresholds)(eer))
    return eer,thresh


def evaluate(model, gallery_loader, probe_loader, device, out_dir, tag):
    gal_feats,gal_labels=extract_embeddings(model,gallery_loader,device)
    prb_feats,prb_labels=extract_embeddings(model,probe_loader,device)
    sim  =prb_feats@gal_feats.T
    rank1=100.0*(gal_labels[sim.argmax(axis=1)]==prb_labels).mean()
    scores_list,labels_list=[],[]
    for i in range(len(prb_labels)):
        for j in range(len(gal_labels)):
            scores_list.append(float(sim[i,j]))
            labels_list.append(1 if prb_labels[i]==gal_labels[j] else -1)
    scores_arr=np.column_stack([scores_list,labels_list])
    eer,_=compute_eer(scores_arr)
    os.makedirs(out_dir,exist_ok=True)
    with open(os.path.join(out_dir,f"scores_{tag}.txt"),"w") as f:
        for s,l in zip(scores_list,labels_list): f.write(f"{s} {l}\n")
    print(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer,rank1


# ══════════════════════════════════════════════════════════════
#  SINGLE EXPERIMENT
# ══════════════════════════════════════════════════════════════

def run_experiment(train_data, test_data, cfg, device=None):
    seed=cfg["random_seed"]; results_dir=cfg["results_dir"]
    eval_every=cfg["eval_every"]; cache_dir=cfg["base_results_dir"]
    eval_tag_base=test_data.replace("-","")

    os.makedirs(results_dir,exist_ok=True)
    rst_eval=os.path.join(results_dir,"eval"); os.makedirs(rst_eval,exist_ok=True)

    same_dataset=(_ds_key(train_data)==_ds_key(test_data))

    if same_dataset:
        print(f"  Parsing {train_data} (shared train+test) …")
        all_id2paths=get_parser(train_data,cfg)()
        (train_samples,gallery_samples,probe_samples,
         train_label_map,_)=split_same_dataset(
            all_id2paths,cfg["train_subject_ratio"],cfg["test_gallery_ratio"],seed)
        num_classes=len(train_label_map)
    else:
        print(f"  Parsing {train_data} (train) …")
        train_id2paths=get_parser(train_data,cfg)()
        train_label_map={k:i for i,k in enumerate(sorted(train_id2paths))}
        train_samples=[(p,train_label_map[ident])
                       for ident,paths in train_id2paths.items() for p in paths]
        num_classes=len(train_label_map)
        print(f"  Parsing {test_data} (test) …")
        test_id2paths=get_parser(test_data,cfg)()
        gallery_samples,probe_samples,_=split_cross_dataset_test(
            test_id2paths,cfg["test_gallery_ratio"],seed)

    train_loader  =make_train_loader(train_samples,cfg)
    gallery_loader=make_eval_loader(gallery_samples,cfg)
    probe_loader  =make_eval_loader(probe_samples,cfg)
    print(f"  Gallery={len(gallery_samples)}  Probe={len(probe_samples)}  Classes={num_classes}")

    model    =MagFaceBackbone(cfg["pretrained_weights"]).to(device)
    criterion=MagFaceLoss(num_classes,embedding_size=512,
                          s=cfg["arcface_s"],m_l=cfg["m_l"],m_u=cfg["m_u"],
                          l_a=cfg["l_a"],u_a=cfg["u_a"],
                          lambda_g=cfg["lambda_g"]).to(device)

    get_or_create_init_weights(model,criterion,num_classes,cache_dir,device)

    trainable_params=([p for p in model.parameters()     if p.requires_grad]+
                      [p for p in criterion.parameters() if p.requires_grad])
    optimizer=optim.AdamW(trainable_params,lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg["num_epochs"],eta_min=1e-6)

    best_rank1=0.0; ckpt_path=os.path.join(results_dir,"best_model.pth")

    evaluate(model,gallery_loader,probe_loader,device,rst_eval,f"pretrain_{eval_tag_base}")

    for epoch in range(1,cfg["num_epochs"]+1):
        model.train(); criterion.train()
        ep_loss=0.0; ep_arc=0.0; ep_g=0.0; ep_norm=0.0; ep_corr=0; ep_tot=0

        for imgs,labels in train_loader:
            imgs=imgs.to(device); labels=labels.to(device)
            optimizer.zero_grad()
            embeddings=model(imgs)
            loss,l_arc,l_g=criterion(embeddings,labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params,5.0)
            optimizer.step()
            ep_loss+=loss.item(); ep_arc+=l_arc; ep_g+=l_g
            ep_norm+=embeddings.norm(dim=1).mean().item()
            with torch.no_grad():
                preds=criterion.get_logits(embeddings).argmax(dim=1)
                ep_corr+=(preds==labels).sum().item(); ep_tot+=labels.size(0)

        scheduler.step()
        n=len(train_loader); acc=100.*ep_corr/max(ep_tot,1)
        ts=time.strftime("%H:%M:%S")
        print(f"  [{ts}] ep {epoch:03d}/{cfg['num_epochs']}  "
              f"loss={ep_loss/n:.4f}  arc={ep_arc/n:.4f}  "
              f"L_g={ep_g/n:.4f}  norm={ep_norm/n:.2f}  acc={acc:.2f}%")

        if epoch%eval_every==0 or epoch==cfg["num_epochs"]:
            cur_eer,cur_rank1=evaluate(model,gallery_loader,probe_loader,device,
                                       rst_eval,f"ep{epoch:04d}_{eval_tag_base}")
            if cur_rank1>best_rank1:
                best_rank1=cur_rank1
                torch.save({"epoch":epoch,"model":model.state_dict(),
                            "criterion":criterion.state_dict(),
                            "rank1":cur_rank1,"eer":cur_eer},ckpt_path)
                print(f"  *** New best Rank-1: {best_rank1:.2f}% ***")

    ckpt=torch.load(ckpt_path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_eer,final_rank1=evaluate(model,gallery_loader,probe_loader,device,
                                   rst_eval,f"FINAL_{eval_tag_base}")
    return final_eer,final_rank1


# ══════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ══════════════════════════════════════════════════════════════

def print_and_save_table(results, train_datasets, test_datasets, out_path):
    col_w=14; td_label=[t.replace("-","") for t in test_datasets]+["Avg"]
    header=f"{'Train\\Test':<14}"+"".join(f"{t:>{col_w}}" for t in td_label)
    sep="─"*len(header); lines=[]
    for metric_label,idx in [("EER (%)",0),("Rank-1 (%)",1)]:
        lines.append(f"\n{metric_label} Results")
        lines.append(sep); lines.append(header); lines.append(sep)
        for tr in train_datasets:
            row=f"{tr.replace('-',''):<14}"; vals=[]
            for te in test_datasets:
                val=results.get((tr,te))
                cell=f"{val[idx]:.2f}" if val is not None else "—"
                row+=f"{cell:>{col_w}}"
                if val is not None: vals.append(val[idx])
            avg_cell=f"{sum(vals)/len(vals):.2f}" if vals else "—"
            row+=f"{avg_cell:>{col_w}}"; lines.append(row)
        lines.append(sep)
    text="\n".join(lines); print(text)
    with open(out_path,"w") as f: f.write(text+"\n")
    print(f"\nTable saved to: {out_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    seed=BASE_CONFIG["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir=BASE_CONFIG["base_results_dir"]
    os.makedirs(base_results_dir,exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MagFace iResNet100 — Full Cross-Dataset Experiment")
    print(f"  Device : {device}")
    print(f"  Train  : {TRAIN_DATASETS}")
    print(f"  Test   : {TEST_DATASETS}")
    print(f"  Epochs : {BASE_CONFIG['num_epochs']}")
    print(f"{'='*60}\n")

    n_total=len(TRAIN_DATASETS)*len(TEST_DATASETS); n_done=0; results={}; failures=[]

    for train_data in TRAIN_DATASETS:
        for test_data in TEST_DATASETS:
            n_done+=1
            exp_label=f"train={train_data}  test={test_data}"
            print(f"\n{'='*60}")
            print(f"  Experiment {n_done}/{n_total}:  {exp_label}")
            print(f"{'='*60}")

            cfg=copy.deepcopy(BASE_CONFIG)
            cfg["train_data"]=train_data; cfg["test_data"]=test_data
            safe_train=train_data.replace("-","").replace(" ","")
            safe_test =test_data.replace("-","").replace(" ","")
            cfg["results_dir"]=os.path.join(base_results_dir,f"train_{safe_train}_test_{safe_test}")

            t_start=time.time()
            try:
                eer,rank1=run_experiment(train_data,test_data,cfg,device=device)
                results[(train_data,test_data)]=(eer*100,rank1)
                elapsed=time.time()-t_start
                print(f"\n  ✓  {exp_label}")
                print(f"     EER={eer*100:.4f}%  Rank-1={rank1:.2f}%  Time={elapsed/60:.1f} min")
            except Exception as e:
                results[(train_data,test_data)]=None
                failures.append((train_data,test_data,str(e)))
                print(f"\n  ✗  {exp_label}  FAILED: {e}")

    table_path=os.path.join(base_results_dir,"results_table.txt")
    print(f"\n\n{'='*60}"); print(f"  ALL EXPERIMENTS COMPLETE"); print(f"{'='*60}")
    print_and_save_table(results,TRAIN_DATASETS,TEST_DATASETS,table_path)

    if failures:
        print(f"\nFailed ({len(failures)}):")
        for tr,te,err in failures: print(f"  train={tr}  test={te}  → {err}")

    json_results={f"{tr}→{te}":list(v) if v else None for (tr,te),v in results.items()}
    with open(os.path.join(base_results_dir,"results_raw.json"),"w") as f:
        json.dump(json_results,f,indent=2)
    print(f"\nRaw results: {os.path.join(base_results_dir,'results_raw.json')}")


if __name__ == "__main__":
    main()
