"""
ArcFace iResNet100 — Cross-Domain Open-Set Evaluations on X-Palm
=====================================================================
Follows the exact same evaluation framework as CompNet open-set.

Settings (12 total)
────────────────────
  S_scanner         │ Train : perspective (all)  for train IDs
                    │ Test  : scanner            for test IDs

  S_scanner_to_persp│ Train : scanner            for scanner IDs
                    │ Test  : perspective (all)  for IDs with NO scanner data

  S_(A,B) (×10)     │ Train : perspective(¬A,¬B) + scanner  for train IDs
                    │ Test  : 1 img A → gallery / 1 img B → probe, test IDs

Shared splits file: x_palm_openset_splits.json
"""

# ==============================================================
#  CONFIG
# ==============================================================
CONFIG = {
    "x_palm_data_root"  : "path to x-palm data",
    "scanner_spectra"      : {"green", "ir", "yellow", "pink", "white"},

    "train_id_ratio"       : 0.80,
    "test_gallery_ratio"   : 0.50,

    "pretrained_weights"   : "./checkpoints/r100_glint360k.onnx",

    "arcface_s"            : 64.0,
    "arcface_m"            : 0.50,

    "img_side"             : 112,
    "batch_size"           : 32,
    "num_epochs"           : 200,
    "lr"                   : 1e-4,
    "weight_decay"         : 5e-4,
    "eval_every"           : 5,
    "num_workers"          : 4,

    "base_results_dir"     : "./rst_arcface_crossdomain_openset",
    "random_seed"          : 42,
}

PAIRED_CONDITIONS = [
    ("wet",  "text"), ("wet",  "rnd"),  ("rnd",  "text"),
    ("sf",   "roll"), ("jf",   "pitch"), ("bf",   "far"),
    ("roll", "close"), ("far", "jf"),   ("fl",   "sf"),
    ("roll", "pitch"),
]

SPLITS_FILE = "./x_palm_openset_splits.json"
# ==============================================================

import os, json, math, time, random, warnings
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
IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


# ══════════════════════════════════════════════════════════════
#  MODEL  (unchanged)
# ══════════════════════════════════════════════════════════════

class ArcFaceBackbone(nn.Module):
    def __init__(self, pretrained_path, freeze_ratio=0.75):
        super().__init__()
        import onnx
        from onnx2torch import convert
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"ONNX not found: {pretrained_path}")
        print(f"  Loading ONNX model: {pretrained_path}")
        self.net = convert(onnx.load(pretrained_path))
        all_params = list(self.net.parameters())
        n_freeze   = int(len(all_params) * freeze_ratio)
        for i, p in enumerate(all_params):
            p.requires_grad = (i >= n_freeze)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (list, tuple)): out = out[0]
        return F.normalize(out, p=2, dim=1)


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=512, s=64.0, m=0.50):
        super().__init__()
        self.s=s; self.cos_m=math.cos(m); self.sin_m=math.sin(m)
        self.th=math.cos(math.pi-m); self.mm=math.sin(math.pi-m)*m
        self.weight=nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight); self.ce=nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        W=F.normalize(self.weight,p=2,dim=1)
        cos_theta=(embeddings@W.t()).clamp(-1+1e-7,1-1e-7)
        sin_theta=(1.0-cos_theta**2).sqrt()
        cos_theta_m=cos_theta*self.cos_m-sin_theta*self.sin_m
        cos_theta_m=torch.where(cos_theta>self.th,cos_theta_m,cos_theta-self.mm)
        one_hot=torch.zeros_like(cos_theta).scatter_(1,labels.view(-1,1),1.0)
        logits=self.s*(one_hot*cos_theta_m+(1.0-one_hot)*cos_theta)
        return self.ce(logits,labels)

    @torch.no_grad()
    def get_logits(self, embeddings):
        W=F.normalize(self.weight,p=2,dim=1)
        return self.s*(embeddings@W.t())


# ══════════════════════════════════════════════════════════════
#  TRANSFORMS & DATASETS
# ══════════════════════════════════════════════════════════════

def _base_tf(img_side):
    return transforms.Compose([
        transforms.Resize((img_side,img_side)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

def _aug_tf(img_side):
    return transforms.Compose([
        transforms.Resize((img_side,img_side)),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0,contrast=0.05,saturation=0,hue=0),
            transforms.RandomResizedCrop(img_side,scale=(0.8,1.0),ratio=(1.0,1.0)),
            transforms.RandomPerspective(distortion_scale=0.15,p=1),
            transforms.RandomChoice([
                transforms.RandomRotation(10,interpolation=Image.BICUBIC,
                                          expand=False,center=(int(0.5*img_side),0)),
                transforms.RandomRotation(10,interpolation=Image.BICUBIC,
                                          expand=False,center=(0,int(0.5*img_side))),
            ]),
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

class TrainDataset(Dataset):
    def __init__(self,samples,img_side):
        self.samples=samples; self.transform=_aug_tf(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        path,label=self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")),label

class EvalDataset(Dataset):
    def __init__(self,samples,img_side):
        self.samples=samples; self.transform=_base_tf(img_side)
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        path,label=self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")),label

def make_loader(samples,train,cfg):
    ds=TrainDataset(samples,cfg["img_side"]) if train else EvalDataset(samples,cfg["img_side"])
    return DataLoader(ds,batch_size=min(cfg["batch_size"],len(samples)),
                      shuffle=train,num_workers=cfg["num_workers"],pin_memory=True,
                      drop_last=train and len(samples)>cfg["batch_size"])


# ══════════════════════════════════════════════════════════════
#  DATA COLLECTION HELPERS  (identical to CompNet)
# ══════════════════════════════════════════════════════════════

def _collect_perspective(data_root):
    cond_paths=defaultdict(lambda: defaultdict(list))
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir=os.path.join(data_root,subject_id)
        if not os.path.isdir(subject_dir): continue
        roi_dir=os.path.join(subject_dir,"roi_perspective")
        if not os.path.isdir(roi_dir): continue
        for fname in sorted(os.listdir(roi_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts=os.path.splitext(fname)[0].split("_")
            if len(parts)<3: continue
            cond_paths[parts[2].lower()][parts[0]+"_"+parts[1].lower()].append(
                os.path.join(roi_dir,fname))
    return cond_paths

def _collect_scanner(data_root,scanner_spectra):
    scanner_paths=defaultdict(list)
    for subject_id in sorted(os.listdir(data_root)):
        subject_dir=os.path.join(data_root,subject_id)
        if not os.path.isdir(subject_dir): continue
        scan_dir=os.path.join(subject_dir,"roi_scanner")
        if not os.path.isdir(scan_dir): continue
        for fname in sorted(os.listdir(scan_dir)):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS: continue
            parts=os.path.splitext(fname)[0].split("_")
            if len(parts)<4: continue
            if parts[2].lower() not in scanner_spectra: continue
            scanner_paths[parts[0]+"_"+parts[1].lower()].append(os.path.join(scan_dir,fname))
    return scanner_paths

def _all_samples(id2paths,label_map):
    return [(p,label_map[ident]) for ident,paths in id2paths.items() for p in paths]

def _gallery_probe_split(id2paths,label_map,gallery_ratio,rng):
    gallery,probe=[],[]
    for ident,paths in id2paths.items():
        paths=list(paths); rng.shuffle(paths)
        n_gal=max(1,int(len(paths)*gallery_ratio))
        n_gal=min(n_gal,len(paths)-1)
        if len(paths)==1:
            gallery.append((paths[0],label_map[ident]))
            probe.append((paths[0],label_map[ident]))
        else:
            for p in paths[:n_gal]: gallery.append((p,label_map[ident]))
            for p in paths[n_gal:]: probe.append((p,label_map[ident]))
    return gallery,probe


# ══════════════════════════════════════════════════════════════
#  TRAIN/TEST ID SPLIT PERSISTENCE  (identical to CompNet)
# ══════════════════════════════════════════════════════════════

def generate_all_splits(cond_paths,scanner_paths,train_id_ratio,seed):
    import random as _random
    persp_all=defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident,paths in cond_dict.items():
            persp_all[ident].extend(paths)
    all_persp_ids=sorted(persp_all.keys()); scanner_ids=sorted(scanner_paths.keys())
    n_test=len(all_persp_ids)-int(len(all_persp_ids)*train_id_ratio)
    splits={}
    test_ids=sorted(_random.Random(seed).sample(scanner_ids,n_test))
    train_ids=sorted(set(all_persp_ids)-set(test_ids))
    splits["S_scanner"]={"train_ids":train_ids,"test_ids":test_ids}
    no_scanner_ids=sorted(set(all_persp_ids)-set(scanner_ids))
    splits["S_scanner_to_persp"]={"train_ids":scanner_ids,"test_ids":no_scanner_ids}
    for cond_a,cond_b in PAIRED_CONDITIONS:
        paths_a=cond_paths.get(cond_a,{}); paths_b=cond_paths.get(cond_b,{})
        eligible_ids=sorted(set(paths_a.keys())&set(paths_b.keys()))
        if not eligible_ids: continue
        n_t=min(n_test,len(eligible_ids))
        test_ids=sorted(_random.Random(seed).sample(eligible_ids,n_t))
        train_ids=sorted(set(all_persp_ids)-set(test_ids))
        splits[f"S_{cond_a}_{cond_b}"]={"train_ids":train_ids,"test_ids":test_ids}
    return splits

def load_or_generate_splits(cond_paths,scanner_paths,train_id_ratio,seed):
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE) as f: splits=json.load(f)
        print(f"  Loaded existing ID splits from: {SPLITS_FILE}")
        for key,val in splits.items():
            print(f"    {key:<30}  train={len(val['train_ids'])}  test={len(val['test_ids'])}")
    else:
        print(f"  Generating ID splits (seed={seed}) → {SPLITS_FILE}")
        splits=generate_all_splits(cond_paths,scanner_paths,train_id_ratio,seed)
        with open(SPLITS_FILE,"w") as f: json.dump(splits,f,indent=2)
        print(f"  Splits saved to: {SPLITS_FILE}")
        for key,val in splits.items():
            print(f"    {key:<30}  train={len(val['train_ids'])}  test={len(val['test_ids'])}")
    return splits


# ══════════════════════════════════════════════════════════════
#  PARSERS  (identical to CompNet)
# ══════════════════════════════════════════════════════════════

def parse_setting_scanner(cond_paths,scanner_paths,splits,gallery_ratio,seed):
    rng=random.Random(seed)
    persp_all=defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident,paths in cond_dict.items(): persp_all[ident].extend(paths)
    train_ids=splits["train_ids"]; test_ids=splits["test_ids"]
    train_label_map={ident:i for i,ident in enumerate(train_ids)}
    test_label_map ={ident:i for i,ident in enumerate(test_ids)}
    train_samples=_all_samples({i:persp_all[i] for i in train_ids if i in persp_all},train_label_map)
    gallery,probe=_gallery_probe_split(
        {i:scanner_paths[i] for i in test_ids if i in scanner_paths},test_label_map,gallery_ratio,rng)
    _print_stats("S_scanner | Perspective (train IDs) → Scanner (test IDs)",
                 len(train_ids),len(test_ids),len(train_samples),len(gallery),len(probe))
    return train_samples,gallery,probe,len(train_ids)

def parse_setting_scanner_to_perspective(cond_paths,scanner_paths,splits,gallery_ratio,seed):
    rng=random.Random(seed)
    persp_all=defaultdict(list)
    for cond_dict in cond_paths.values():
        for ident,paths in cond_dict.items(): persp_all[ident].extend(paths)
    train_ids=splits["train_ids"]; test_ids=splits["test_ids"]
    train_label_map={ident:i for i,ident in enumerate(train_ids)}
    test_label_map ={ident:i for i,ident in enumerate(test_ids)}
    train_samples=_all_samples(scanner_paths,train_label_map)
    gallery,probe=_gallery_probe_split(
        {i:persp_all[i] for i in test_ids if i in persp_all},test_label_map,gallery_ratio,rng)
    _print_stats("S_scanner_to_persp | Scanner (all) → Perspective (no-scanner IDs)",
                 len(train_ids),len(test_ids),len(train_samples),len(gallery),len(probe))
    return train_samples,gallery,probe,len(train_ids)

def parse_setting_paired_conditions(cond_a,cond_b,cond_paths,scanner_paths,splits,seed):
    rng=random.Random(seed)
    paths_a=cond_paths.get(cond_a,{}); paths_b=cond_paths.get(cond_b,{})
    train_ids=splits["train_ids"]; test_ids=splits["test_ids"]
    train_label_map={ident:i for i,ident in enumerate(train_ids)}
    test_label_map ={ident:i for i,ident in enumerate(test_ids)}
    train_samples=[]
    for cond,cond_dict in cond_paths.items():
        if cond in (cond_a,cond_b): continue
        for ident in train_ids:
            for p in cond_dict.get(ident,[]): train_samples.append((p,train_label_map[ident]))
    for ident in train_ids:
        for p in scanner_paths.get(ident,[]): train_samples.append((p,train_label_map[ident]))
    gallery_samples,probe_samples=[],[]
    for ident in test_ids:
        label=test_label_map[ident]
        a_imgs=list(paths_a.get(ident,[])); rng.shuffle(a_imgs)
        b_imgs=list(paths_b.get(ident,[])); rng.shuffle(b_imgs)
        if not a_imgs or not b_imgs: continue
        if rng.random()<0.5:
            gallery_samples.append((a_imgs[0],label)); probe_samples.append((b_imgs[0],label))
        else:
            gallery_samples.append((b_imgs[0],label)); probe_samples.append((a_imgs[0],label))
    _print_stats(f"S_{cond_a}_{cond_b} | Perspective(not {cond_a}/{cond_b})+Scanner → {cond_a}/{cond_b}",
                 len(train_ids),len(test_ids),len(train_samples),len(gallery_samples),len(probe_samples))
    return train_samples,gallery_samples,probe_samples,len(train_ids)

def _print_stats(name,n_train_ids,n_test_ids,train_n,gallery_n,probe_n):
    print(f"\n  [{name}]")
    print(f"    Train IDs / Test IDs  : {n_train_ids} / {n_test_ids}")
    print(f"    Train images          : {train_n}")
    print(f"    Gallery / Probe       : {gallery_n} / {probe_n}")


# ══════════════════════════════════════════════════════════════
#  FIXED MODEL INITIALISATION
# ══════════════════════════════════════════════════════════════

def get_or_create_init_weights(model,criterion,num_classes,cache_dir,device):
    os.makedirs(cache_dir,exist_ok=True)
    weights_path=os.path.join(cache_dir,f"init_weights_ArcFace_nc{num_classes}.pth")
    if os.path.exists(weights_path):
        print(f"  Loading cached init weights: {weights_path}")
        ckpt=torch.load(weights_path,map_location=device,weights_only=False)
        model.load_state_dict(ckpt["model"]); criterion.load_state_dict(ckpt["arc"])
    else:
        print(f"  Saving init weights: {weights_path}")
        torch.save({"model":model.state_dict(),"arc":criterion.state_dict()},weights_path)


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model,loader,device):
    model.eval(); feats,labels=[],[]
    for imgs,lbl in loader:
        emb=model(imgs.to(device))
        emb=torch.nan_to_num(emb,nan=0.0,posinf=0.0,neginf=0.0)
        feats.append(emb.cpu().numpy()); labels.append(lbl.numpy())
    feats=np.concatenate(feats)
    if not np.isfinite(feats).all():
        feats=np.nan_to_num(feats,nan=0.0,posinf=0.0,neginf=0.0)
    return feats,np.concatenate(labels)

def compute_eer(scores_array):
    ins=scores_array[scores_array[:,1]==1,0]; outs=scores_array[scores_array[:,1]==-1,0]
    if len(ins)==0 or len(outs)==0: return 1.0,0.0
    y=np.concatenate([np.ones(len(ins)),np.zeros(len(outs))]); s=np.concatenate([ins,outs])
    if not np.isfinite(s).all() or np.unique(s).size<2: return 1.0,0.0
    fpr,tpr,thresholds=roc_curve(y,s,pos_label=1)
    eer=brentq(lambda x:1.0-x-interp1d(fpr,tpr)(x),0.0,1.0)
    return eer,float(interp1d(fpr,thresholds)(eer))

def evaluate(model,gallery_loader,probe_loader,device,out_dir,tag):
    gal_feats,gal_labels=extract_embeddings(model,gallery_loader,device)
    prb_feats,prb_labels=extract_embeddings(model,probe_loader,device)
    sim=prb_feats@gal_feats.T
    rank1=100.0*(gal_labels[sim.argmax(axis=1)]==prb_labels).mean()
    scores_list,labels_list=[],[]
    for i in range(len(prb_labels)):
        for j in range(len(gal_labels)):
            scores_list.append(float(sim[i,j]))
            labels_list.append(1 if prb_labels[i]==gal_labels[j] else -1)
    scores_arr=np.column_stack([scores_list,labels_list]); eer,_=compute_eer(scores_arr)
    os.makedirs(out_dir,exist_ok=True)
    with open(os.path.join(out_dir,f"scores_{tag}.txt"),"w") as f:
        for s,l in zip(scores_list,labels_list): f.write(f"{s} {l}\n")
    print(f"  [{tag}]  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%")
    return eer,rank1


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════

def run_experiment(train_samples,gallery_samples,probe_samples,num_classes,cfg,results_dir,device):
    os.makedirs(results_dir,exist_ok=True)
    rst_eval=os.path.join(results_dir,"eval"); os.makedirs(rst_eval,exist_ok=True)

    train_loader  =make_loader(train_samples,  True, cfg)
    gallery_loader=make_loader(gallery_samples,False,cfg)
    probe_loader  =make_loader(probe_samples,  False,cfg)

    model    =ArcFaceBackbone(cfg["pretrained_weights"]).to(device)
    criterion=ArcFaceLoss(num_classes,embedding_size=512,
                          s=cfg["arcface_s"],m=cfg["arcface_m"]).to(device)

    get_or_create_init_weights(model,criterion,num_classes,cfg["base_results_dir"],device)

    trainable_params=([p for p in model.parameters()     if p.requires_grad]+
                      [p for p in criterion.parameters() if p.requires_grad])
    optimizer=optim.AdamW(trainable_params,lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg["num_epochs"],eta_min=1e-6)

    best_rank1=0.0; ckpt_path=os.path.join(results_dir,"best_model.pth")

    evaluate(model,gallery_loader,probe_loader,device,rst_eval,"ep-001_pretrain")

    for epoch in range(1,cfg["num_epochs"]+1):
        model.train(); criterion.train()
        ep_loss=0.0; ep_corr=0; ep_tot=0
        for imgs,labels in train_loader:
            imgs=imgs.to(device); labels=labels.to(device)
            optimizer.zero_grad()
            embeddings=model(imgs); loss=criterion(embeddings,labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params,5.0)
            optimizer.step()
            ep_loss+=loss.item()
            with torch.no_grad():
                preds=criterion.get_logits(embeddings).argmax(dim=1)
                ep_corr+=(preds==labels).sum().item(); ep_tot+=labels.size(0)
        scheduler.step()
        n=len(train_loader); acc=100.*ep_corr/max(ep_tot,1); ts=time.strftime("%H:%M:%S")
        if epoch%10==0 or epoch==cfg["num_epochs"]:
            print(f"  [{ts}] ep {epoch:04d} | loss={ep_loss/n:.4f} | acc={acc:.2f}%")
        if epoch%cfg["eval_every"]==0 or epoch==cfg["num_epochs"]:
            cur_eer,cur_rank1=evaluate(model,gallery_loader,probe_loader,device,rst_eval,f"ep{epoch:04d}")
            if cur_rank1>best_rank1:
                best_rank1=cur_rank1
                torch.save({"epoch":epoch,"model":model.state_dict(),
                            "arc":criterion.state_dict(),"rank1":cur_rank1,"eer":cur_eer},ckpt_path)
                print(f"  *** New best Rank-1: {best_rank1:.2f}% ***")

    ckpt=torch.load(ckpt_path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_eer,final_rank1=evaluate(model,gallery_loader,probe_loader,device,rst_eval,"FINAL")
    return final_eer,final_rank1


# ══════════════════════════════════════════════════════════════
#  RESULTS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════

def print_and_save_summary(all_results,out_path):
    col_w=14
    header=(f"{'Setting':<22}{'Train domain':<38}{'Test domain':<26}"
            f"{'EER (%)':>{col_w}}{'Rank-1 (%)':>{col_w}}")
    sep="─"*len(header)
    lines=["\nCross-Domain Open-Set Results — Palm-Auth (ArcFace iResNet100)",sep,header,sep]
    for r in all_results:
        eer_str  =f"{r['eer']:.2f}"   if r['eer']   is not None else "—"
        rank1_str=f"{r['rank1']:.2f}" if r['rank1'] is not None else "—"
        lines.append(f"{r['setting']:<22}{r['train_desc']:<38}{r['test_desc']:<26}"
                     f"{eer_str:>{col_w}}{rank1_str:>{col_w}}")
    lines.append(sep)
    text="\n".join(lines); print(text)
    with open(out_path,"w") as f: f.write(text+"\n")
    print(f"\nSummary saved to: {out_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    cfg=CONFIG; seed=cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_results_dir=cfg["base_results_dir"]; os.makedirs(base_results_dir,exist_ok=True)
    train_ratio=cfg["train_id_ratio"]; gallery_ratio=cfg["test_gallery_ratio"]

    print(f"\n{'='*60}")
    print(f"  ArcFace iResNet100 — Cross-Domain Open-Set (Palm-Auth)")
    print(f"  Protocol  : open set ({train_ratio*100:.0f}/{(1-train_ratio)*100:.0f} ID split)")
    print(f"  Device    : {device}  Epochs : {cfg['num_epochs']}")
    print(f"  Settings  : 2 scanner + {len(PAIRED_CONDITIONS)} paired-condition")
    print(f"  Results   : {base_results_dir}")
    print(f"{'='*60}")

    print("\n  Scanning dataset …")
    cond_paths   =_collect_perspective(cfg["palm_auth_data_root"])
    scanner_paths=_collect_scanner(cfg["palm_auth_data_root"],cfg["scanner_spectra"])
    print(f"  Perspective conditions : {sorted(cond_paths.keys())}")
    print(f"  Scanner identities     : {len(scanner_paths)}")

    all_splits=load_or_generate_splits(cond_paths,scanner_paths,train_ratio,seed)

    SETTINGS=[]
    SETTINGS.append({"tag":"setting_scanner","label":"S_scanner",
                     "train_desc":"Perspective (train IDs)","test_desc":"Scanner (test IDs)",
                     "parser":lambda: parse_setting_scanner(
                         cond_paths,scanner_paths,all_splits["S_scanner"],gallery_ratio,seed)})
    SETTINGS.append({"tag":"setting_scanner_to_persp","label":"S_scanner_to_persp",
                     "train_desc":"Scanner (all scanner IDs)","test_desc":"Perspective (no-scanner IDs)",
                     "parser":lambda: parse_setting_scanner_to_perspective(
                         cond_paths,scanner_paths,all_splits["S_scanner_to_persp"],gallery_ratio,seed)})

    conditions_found=sorted(cond_paths.keys())
    for cond_a,cond_b in PAIRED_CONDITIONS:
        if cond_a not in conditions_found or cond_b not in conditions_found:
            print(f"  [WARN] '{cond_a}' or '{cond_b}' not found — skipping"); continue
        ca,cb=cond_a,cond_b; split_key=f"S_{ca}_{cb}"
        if split_key not in all_splits:
            print(f"  [WARN] No split for {split_key} — skipping"); continue
        SETTINGS.append({
            "tag":f"setting_{ca}_{cb}","label":f"S_{ca}_{cb}",
            "train_desc":f"Perspective(not {ca}/{cb})+Scanner","test_desc":f"{ca}/{cb} (test IDs)",
            "parser":(lambda ca=ca,cb=cb: parse_setting_paired_conditions(
                ca,cb,cond_paths,scanner_paths,all_splits[f"S_{ca}_{cb}"],seed))})

    print(f"\n  Total settings to run : {len(SETTINGS)}")
    all_results=[]

    for idx,s in enumerate(SETTINGS,1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{len(SETTINGS)}] {s['label']}")
        print(f"  Train : {s['train_desc']}"); print(f"  Test  : {s['test_desc']}")
        print(f"{'='*60}")
        results_dir=os.path.join(base_results_dir,s["tag"]); t_start=time.time()
        try:
            train_s,gal_s,probe_s,n_cls=s["parser"]()
            eer,rank1=run_experiment(train_s,gal_s,probe_s,n_cls,cfg,results_dir,device)
            elapsed=time.time()-t_start
            print(f"\n  ✓  {s['label']}:  EER={eer*100:.4f}%  Rank-1={rank1:.2f}%  Time={elapsed/60:.1f} min")
            with open(os.path.join(results_dir,"results.json"),"w") as f:
                json.dump({"setting":s["label"],"train_desc":s["train_desc"],
                           "test_desc":s["test_desc"],"num_train_classes":n_cls,
                           "EER_pct":eer*100,"Rank1_pct":rank1},f,indent=2)
            all_results.append({"setting":s["label"],"train_desc":s["train_desc"],
                                 "test_desc":s["test_desc"],"eer":eer*100,"rank1":rank1})
        except Exception as e:
            print(f"\n  ✗  {s['label']} FAILED: {e}")
            all_results.append({"setting":s["label"],"train_desc":s["train_desc"],
                                 "test_desc":s["test_desc"],"eer":None,"rank1":None})

    print(f"\n\n{'='*60}"); print(f"  ALL {len(SETTINGS)} SETTINGS COMPLETE"); print(f"{'='*60}")
    print_and_save_summary(all_results,os.path.join(base_results_dir,"results_summary.txt"))


if __name__ == "__main__":
    main()
