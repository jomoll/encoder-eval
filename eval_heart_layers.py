#!/usr/bin/env python3
"""
Layer-wise accessibility analysis for CLIP-like models (ViT & ResNet)
=====================================================================

Given a (fine-tuned or pretrained) CLIP-like model and a dataset (e.g., jomoll/silent-heart-dataset),
this script extracts intermediate features across layers, applies several readouts, and fits probes
(linear, MLP, and optionally Random Fourier Features) to measure accessibility of a factor (e.g., heart).

Backbones
---------
- ViT (HuggingFace CLIP ViT): uses vision_model(..., output_hidden_states=True)
- ResNet (your custom wrappers from train_clip_modes.py): hooks into layer1..4 of torchvision resnet

Readouts
--------
ViT:    cls, mean_tokens, region   (region uses 'special_object' metadata if present)
ResNet: gap, gmp, region

Probes
------
- linear (logistic regression head)
- mlp    (1 hidden layer, small width)
- rff    (Random Fourier Features + linear head)   [optional, --use_rff]

Task labels
-----------
--task ∈ {heart, triangle, both}. Heart uses 'H' or id suffix; triangle scans named_objects; both runs heart and triangle separately.

Usage (example)
---------------
python layerwise_accessibility.py \
  --dataset_id jomoll/silent-heart-dataset-p00 \
  --model_path ./ckpt_slip_vitb32/best \
  --task both \
  --backbone auto \
  --splits train val \
  --vit_layers 2 4 6 8 10 12 \
  --resnet_stages layer1 layer2 layer3 layer4 \
  --readouts_vit cls mean_tokens region \
  --readouts_resnet gap gmp region \
  --probes linear mlp \
  --batch_size 512 --epochs 20 \
  --save_dir ./results/layerwise_heart

Notes
-----
- Region pooling needs 'special_object' (cx,cy,size,angle) in HR coords and 'HR' scalar.
- If metadata missing or maps out of bounds, region readout falls back to global mean.
"""

import os, argparse, json, math, random, hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from datasets import load_dataset
from transformers import CLIPModel, CLIPImageProcessor

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

# Try to import your custom wrappers (for ResNet)
try:
    from train_clip_modes import ResNet18CLIP, DenseNet121CLIP, VGG11CLIP, SmallResNetCLIP, TinyResNetCLIP
except Exception:
    ResNet18CLIP = DenseNet121CLIP = VGG11CLIP = SmallResNetCLIP = TinyResNetCLIP = None

# ----------------------------
# Basic utils
# ----------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def caption_id_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

# ----------------------------
# Transforms (eval-style)
# ----------------------------

class EvalTransform:
    def __init__(self, image_size: int, mean, std):
        self.t = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    def __call__(self, img: Image.Image):
        if img.mode != "RGB": img = img.convert("RGB")
        return self.t(img)

# ----------------------------
# Labels from dataset
# ----------------------------

def parse_json_field(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return {}
    return x if isinstance(x, dict) else {}

def extract_named_list(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ["named_objects","named","named_captions"]:
        if key in ex and ex[key] is not None:
            v = ex[key]
            if isinstance(v, str):
                try: return json.loads(v)
                except Exception: pass
            if isinstance(v, list): return v
    if "metadata" in ex and isinstance(ex["metadata"], (dict,str)):
        m = ex["metadata"]
        if isinstance(m, str):
            try: m=json.loads(m)
            except: m={}
        if isinstance(m, dict):
            for key in ["named_objects","named","named_captions"]:
                if key in m and m[key] is not None:
                    v = m[key]
                    if isinstance(v, str):
                        try: return json.loads(v)
                        except Exception: pass
                    if isinstance(v, list): return v
    return []

def has_triangle_named(ex: Dict[str, Any]) -> int:
    try:
        for o in extract_named_list(ex):
            if str(o.get("shape","")).lower() == "triangle": return 1
    except Exception:
        pass
    return 0

def resolve_label(ex: Dict[str, Any], task: str) -> int:
    if task == "heart":
        if "H" in ex:
            v = ex["H"]; 
            try: return int(v) if isinstance(v, int) else (1 if str(v).lower() in ["1","heart","true","yes"] else 0)
            except: return 0
        if "id" in ex and isinstance(ex["id"], str) and ex["id"][-2:] in ["_0","_1"]:
            return int(ex["id"][-1])
        if "metadata" in ex and isinstance(ex["metadata"], dict) and "H" in ex["metadata"]:
            return int(ex["metadata"]["H"])
        raise ValueError("Cannot resolve heart label: need 'H' or id suffix _0/_1.")
    elif task == "triangle":
        return has_triangle_named(ex)
    else:
        raise ValueError(f"Unknown task {task}")

def resolve_both_labels(ex: Dict[str, Any]) -> Dict[str, int]:
    """Resolve both heart and triangle labels for a single example."""
    labels = {}
    
    # Heart label
    if "H" in ex:
        v = ex["H"]; 
        try: labels["heart"] = int(v) if isinstance(v, int) else (1 if str(v).lower() in ["1","heart","true","yes"] else 0)
        except: labels["heart"] = 0
    elif "id" in ex and isinstance(ex["id"], str) and ex["id"][-2:] in ["_0","_1"]:
        labels["heart"] = int(ex["id"][-1])
    elif "metadata" in ex and isinstance(ex["metadata"], dict) and "H" in ex["metadata"]:
        labels["heart"] = int(ex["metadata"]["H"])
    else:
        labels["heart"] = 0  # Default for both task when heart label missing
    
    # Triangle label
    labels["triangle"] = has_triangle_named(ex)
    
    return labels

# ----------------------------
# Region mapping
# ----------------------------

def region_center_xy(ex, out_W, out_H) -> Optional[Tuple[int,int]]:
    so = ex.get("special_object", None)
    if so is None: return None
    if isinstance(so, str):
        try: so = json.loads(so)
        except Exception: return None
    try:
        HR = float(ex.get("HR", 448))
        cx = float(so["cx"]) / HR * out_W
        cy = float(so["cy"]) / HR * out_H
        return int(round(cx)), int(round(cy))
    except Exception:
        return None

# ----------------------------
# Dataset wrapper
# ----------------------------

def build_dataset(dataset_id: str, split: str, processor: CLIPImageProcessor):
    ds = load_dataset(dataset_id, split=split)
    image_size = processor.size["shortest_edge"]
    tf = EvalTransform(image_size, processor.image_mean, processor.image_std)
    def _t(examples):
        imgs = examples["image"] if isinstance(examples["image"], list) else [examples["image"]]
        is_batched = isinstance(examples["image"], list)
        
        if build_dataset.task == "both":
            out = {"pixel_values": [], "heart_label": [], "triangle_label": [], "id": [], "HR": [], "special_object": []}
            for i, img in enumerate(imgs):
                ex = {k:(examples[k][i] if is_batched else examples[k]) for k in examples.keys()}
                out["pixel_values"].append(tf(img))
                labels = resolve_both_labels(ex)
                out["heart_label"].append(labels["heart"])
                out["triangle_label"].append(labels["triangle"])
                out["id"].append(ex.get("id",""))
                out["HR"].append(ex.get("HR", 448))
                out["special_object"].append(ex.get("special_object", {}))
        else:
            out = {"pixel_values": [], "label": [], "id": [], "HR": [], "special_object": []}
            for i, img in enumerate(imgs):
                ex = {k:(examples[k][i] if is_batched else examples[k]) for k in examples.keys()}
                out["pixel_values"].append(tf(img))
                out["label"].append(resolve_label(ex, build_dataset.task))
                out["id"].append(ex.get("id",""))
                out["HR"].append(ex.get("HR", 448))
                out["special_object"].append(ex.get("special_object", {}))
        return out
    build_dataset.task = None  # set dynamically
    return ds, tf, _t

# ----------------------------
# Probes
# ----------------------------

class LinearProbe(nn.Module):
    def __init__(self, d): super().__init__(); self.fc=nn.Linear(d,2)
    def forward(self,x): return self.fc(x)

class MLPProbe(nn.Module):
    def __init__(self, d, width=None, pdrop=0.1):
        super().__init__()
        if width is None: width = min(512, max(32, int(4*math.sqrt(d))))
        self.net = nn.Sequential(
            nn.Linear(d, width), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(width, 2)
        )
    def forward(self,x): return self.net(x)

class RFFMap(nn.Module):
    """Random Fourier Features for RBF kernel; use linear head on top."""
    def __init__(self, d, num_features=4096, gamma=0.05, seed=0):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        W = torch.randn(d, num_features, generator=g) * math.sqrt(2*gamma)
        b = torch.rand(num_features, generator=g) * 2*math.pi
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.scale = math.sqrt(2.0/num_features)
    def forward(self, x):
        z = x @ self.W + self.b
        return self.scale * torch.cos(z)

class RFFProbe(nn.Module):
    def __init__(self, d, num_features=4096, gamma=0.05, seed=0):
        super().__init__()
        self.map = RFFMap(d, num_features=num_features, gamma=gamma, seed=seed)
        self.fc  = nn.Linear(num_features, 2)
    def forward(self, x): return self.fc(self.map(x))

def train_probe_head(Xtr, ytr, probe: nn.Module, epochs=20, lr=5e-3, wd=0.0, device="cpu"):
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    probe.train()
    bs = min(4096, Xtr.size(0))
    for e in range(epochs):
        perm = torch.randperm(Xtr.size(0), device=device)
        for i in range(0, Xtr.size(0), bs):
            idx = perm[i:i+bs]
            logits = probe(Xtr[idx])
            loss = ce(logits, ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    probe.eval(); return probe

# ----------------------------
# Backbones: feature extraction
# ----------------------------

@dataclass
class FeatureSpec:
    kind: str        # 'vit' or 'resnet'
    layers: List     # vit: list of block indices; resnet: ['layer1',...]
    readouts: List[str]

def is_vit_clip(model) -> bool:
    return isinstance(model, CLIPModel) and hasattr(model, "vision_model") and hasattr(model.vision_model, "embeddings")

def is_custom_resnet(model) -> bool:
    return any(cls is not None and isinstance(model, cls) for cls in [ResNet18CLIP, DenseNet121CLIP, VGG11CLIP, SmallResNetCLIP, TinyResNetCLIP])

def vit_extract_layers(model: CLIPModel, pixel_values: torch.Tensor, target_blocks: List[int]) -> Dict[str, torch.Tensor]:
    """
    Returns dict: key f"vit_block{b}" -> hidden_state [B, seq, D] for that block.
    """
    out = model.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    # hidden_states: list length L+1 (including embeddings output at index 0)
    hs = out.hidden_states  # tuple of [B, seq, D]
    feat = {}
    num_blocks = len(hs) - 1
    for b in target_blocks:
        if b < 0: b = num_blocks + b
        b = max(1, min(num_blocks, b))  # clamp 1..num_blocks
        feat[f"vit_block{b}"] = hs[b]   # [B, seq, D]
    return feat

def vit_readout(h: torch.Tensor, readout: str, ex_batch: List[Dict[str, Any]], grid_hw: Optional[Tuple[int,int]] = None) -> torch.Tensor:
    """
    h: [B, seq, D], seq = 1 + H*W (CLS + patches).
    """
    B, S, D = h.shape
    cls = h[:,0,:]
    patches = h[:,1:,:]
    # infer grid if not given: assume square
    if grid_hw is None:
        HW = S-1
        side = int(round(math.sqrt(HW)))
        Hgrid, Wgrid = side, side
    else:
        Hgrid, Wgrid = grid_hw
    if readout == "cls":
        return cls
    if readout == "mean_tokens":
        return patches.mean(dim=1)
    if readout == "region":
        # region average around special_object center; use 3x3 neighborhood
        patches_2d = patches.reshape(B, Hgrid, Wgrid, D)
        outs = []
        for i in range(B):
            ex = ex_batch[i]
            cxcy = region_center_xy(ex, Wgrid, Hgrid)
            if cxcy is None:
                outs.append(patches_2d[i].mean(dim=(0,1)))  # fallback
                continue
            cx, cy = cxcy
            r = 1
            x0, x1 = max(0, cx-r), min(Wgrid, cx+r+1)
            y0, y1 = max(0, cy-r), min(Hgrid, cy+r+1)
            roi = patches_2d[i, y0:y1, x0:x1, :]
            if roi.numel() == 0:
                outs.append(patches_2d[i].mean(dim=(0,1)))
            else:
                outs.append(roi.mean(dim=(0,1)))
        return torch.stack(outs, dim=0)
    raise ValueError(f"Unknown ViT readout {readout}")

def resnet_register_hooks(backbone, wanted: List[str]):
    feats = {}
    handles=[]
    def hook(name):
        def _fn(m, inp, out): feats[name] = out.detach()
        return _fn
    for n, m in backbone.named_children():
        if n in wanted:
            handles.append(m.register_forward_hook(hook(n)))
    return feats, handles

def resnet_forward_collect(model, pixel_values, stages: List[str]):
    """
    For custom ResNet wrappers: model.vision_model.backbone is torchvision.resnet
    """
    backbone = None
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "backbone"):
        backbone = model.vision_model.backbone
    elif hasattr(model, "vision_model") and isinstance(model.vision_model, nn.Module):
        # try to find a torchvision-like backbone inside
        backbone = getattr(model.vision_model, "backbone", None)
        if backbone is None:
            backbone = model.vision_model
    if backbone is None:
        raise RuntimeError("Cannot locate ResNet backbone for hooks.")
    feats, handles = resnet_register_hooks(backbone, stages)
    # forward through vision_model to trigger hooks
    _ = model.get_image_features(pixel_values=pixel_values) if hasattr(model, "get_image_features") else model(pixel_values=pixel_values, return_loss=False)
    for h in handles: h.remove()
    return feats  # dict stage -> [B,C,H,W]

def resnet_readout(fmap: torch.Tensor, readout: str, ex_batch: List[Dict[str, Any]]) -> torch.Tensor:
    """
    fmap: [B,C,H,W]
    """
    if readout == "gap":
        return fmap.mean(dim=(2,3))
    if readout == "gmp":
        return fmap.amax(dim=(2,3))
    if readout == "region":
        B, C, H, W = fmap.shape
        outs=[]
        for i in range(B):
            ex = ex_batch[i]
            cxcy = region_center_xy(ex, W, H)
            if cxcy is None:
                outs.append(fmap[i].mean(dim=(1,2)))
                continue
            cx, cy = cxcy
            r = max(1, int(round(0.05*max(H,W))))  # small neighborhood
            x0, x1 = max(0, cx-r), min(W, cx+r+1)
            y0, y1 = max(0, cy-r), min(H, cy+r+1)
            roi = fmap[i, :, y0:y1, x0:x1]
            if roi.numel()==0:
                outs.append(fmap[i].mean(dim=(1,2)))
            else:
                outs.append(roi.mean(dim=(1,2)))
        return torch.stack(outs, dim=0)
    raise ValueError(f"Unknown ResNet readout {readout}")

# ----------------------------
# Embedding pass (collect layer→readout features)
# ----------------------------

def collect_features(model, processor, dataset_id, split, task, spec: FeatureSpec,
                     batch_size=512, num_workers=6, device=None, amp_dtype=torch.float16):
    ds = load_dataset(dataset_id, split=split)
    image_size = processor.size["shortest_edge"]
    tf = EvalTransform(image_size, processor.image_mean, processor.image_std)

    def _prep(examples):
        is_b = isinstance(examples["image"], list)
        imgs = examples["image"] if is_b else [examples["image"]]
        
        if task == "both":
            out={"pixel_values":[], "heart_label":[], "triangle_label":[], "id":[], "HR":[], "special_object":[]}
            for i, img in enumerate(imgs):
                ex = {k:(examples[k][i] if is_b else examples[k]) for k in examples.keys()}
                out["pixel_values"].append(tf(img))
                labels = resolve_both_labels(ex)
                out["heart_label"].append(labels["heart"])
                out["triangle_label"].append(labels["triangle"])
                out["id"].append(ex.get("id",""))
                out["HR"].append(ex.get("HR",448))
                out["special_object"].append(ex.get("special_object", {}))
        else:
            out={"pixel_values":[], "label":[], "id":[], "HR":[], "special_object":[]}
            for i, img in enumerate(imgs):
                ex = {k:(examples[k][i] if is_b else examples[k]) for k in examples.keys()}
                out["pixel_values"].append(tf(img))
                out["label"].append(resolve_label(ex, task))
                out["id"].append(ex.get("id",""))
                out["HR"].append(ex.get("HR",448))
                out["special_object"].append(ex.get("special_object", {}))
        return out

    ds = ds.with_transform(_prep)
    
    def _collate(batch):
        if task == "both":
            return {
                "pixel_values": torch.stack([x["pixel_values"] for x in batch], dim=0),
                "heart_label": torch.tensor([x["heart_label"] for x in batch], dtype=torch.long),
                "triangle_label": torch.tensor([x["triangle_label"] for x in batch], dtype=torch.long),
                "ex": batch
            }
        else:
            return {
                "pixel_values": torch.stack([x["pixel_values"] for x in batch], dim=0),
                "label": torch.tensor([x["label"] for x in batch], dtype=torch.long),
                "ex": batch
            }
    
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=(device.type=="cuda"), collate_fn=_collate)
    features = {}  # (layer_key, readout) -> list of tensors
    
    if task == "both":
        heart_labels = []
        triangle_labels = []
    else:
        labels = []
        
    model.eval()
    autocast = torch.cuda.amp.autocast if device.type=="cuda" else torch.cpu.amp.autocast

    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Collect {split}"):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            ex_list = batch["ex"]  # list of dicts for region mapping

            if task == "both":
                y_heart = batch["heart_label"]
                y_triangle = batch["triangle_label"]
            else:
                y = batch["label"]

            with autocast(dtype=amp_dtype):
                if spec.kind == "vit":
                    # run once to get hidden states for chosen blocks
                    out = model.vision_model(pixel_values=pv, output_hidden_states=True, return_dict=True)
                    hs = out.hidden_states  # tuple len L+1
                    num_blocks = len(hs)-1
                    # infer token grid (CLIP ViT-B/32: 7x7 @ 224)
                    HW = hs[-1].shape[1]-1
                    side = int(round(math.sqrt(HW)))
                    grid_hw = (side, side)
                    for b in spec.layers:
                        bb = max(1, min(num_blocks, b))
                        h = hs[bb]  # [B, 1+HW, D]
                        for rd in spec.readouts:
                            feat = vit_readout(h, rd, ex_list, grid_hw=grid_hw)  # [B, D]
                            key = (f"vit_block{bb}", rd)
                            features.setdefault(key, []).append(feat.cpu())
                else:
                    # ResNet: collect stage feature maps via hooks (one forward per batch)
                    stage_feats = resnet_forward_collect(model, pv, spec.layers)  # stage -> [B,C,H,W]
                    for stg in spec.layers:
                        fmap = stage_feats[stg]
                        for rd in spec.readouts:
                            feat = resnet_readout(fmap, rd, ex_list)  # [B, C]
                            key = (stg, rd)
                            features.setdefault(key, []).append(feat.cpu())

            if task == "both":
                heart_labels.append(y_heart)
                triangle_labels.append(y_triangle)
            else:
                labels.append(y)

    feats_np = {k: torch.cat(v, dim=0).numpy() for k, v in features.items()}
    
    if task == "both":
        y_heart_all = torch.cat(heart_labels, dim=0).numpy()
        y_triangle_all = torch.cat(triangle_labels, dim=0).numpy()
        return feats_np, {"heart": y_heart_all, "triangle": y_triangle_all}
    else:
        y_all = torch.cat(labels, dim=0).numpy()
        return feats_np, y_all

# ----------------------------
# Evaluation
# ----------------------------

def evaluate_scores(y_true, scores):
    auc = roc_auc_score(y_true, scores)
    pred = (scores >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred)
    f1  = f1_score(y_true, pred)
    bal = balanced_accuracy_score(y_true, pred)
    return dict(auc=float(auc), acc=float(acc), f1=float(f1), bal_acc=float(bal))

def run_probes(Xtr, ytr, Xev, yev, probes: List[str], device="cpu",
               epochs=20, lr=5e-3, wd=0.0, use_rff=False, rff_dim=4096, rff_gamma=0.05, seed=0):
    res={}
    Xtr_t = torch.from_numpy(Xtr).to(device).float()
    ytr_t = torch.from_numpy(ytr).to(device).long()
    Xev_t = torch.from_numpy(Xev).to(device).float()
    with torch.no_grad():
        # common normalization helps for MLP/RFF
        mu, sigma = Xtr_t.mean(dim=0, keepdim=True), Xtr_t.std(dim=0, keepdim=True).clamp_min(1e-6)
        Xtr_n = (Xtr_t - mu) / sigma
        Xev_n = (Xev_t - mu) / sigma

    if "linear" in probes:
        lin = LinearProbe(Xtr_n.shape[1]).to(device)
        lin = train_probe_head(Xtr_n, ytr_t, lin, epochs=epochs, lr=lr, wd=wd, device=device)
        with torch.no_grad(): s = lin(Xev_n).softmax(dim=1)[:,1].cpu().numpy()
        res["linear"] = evaluate_scores(yev, s)

    if "mlp" in probes:
        mlp = MLPProbe(Xtr_n.shape[1]).to(device)
        mlp = train_probe_head(Xtr_n, ytr_t, mlp, epochs=epochs, lr=lr, wd=wd, device=device)
        with torch.no_grad(): s = mlp(Xev_n).softmax(dim=1)[:,1].cpu().numpy()
        res["mlp"] = evaluate_scores(yev, s)

    if use_rff or ("rff" in probes):
        rff = RFFProbe(Xtr_n.shape[1], num_features=rff_dim, gamma=rff_gamma, seed=seed).to(device)
        rff = train_probe_head(Xtr_n, ytr_t, rff, epochs=epochs, lr=lr, wd=wd, device=device)
        with torch.no_grad(): s = rff(Xev_n).softmax(dim=1)[:,1].cpu().numpy()
        res["rff"] = evaluate_scores(yev, s)

    return res

# ----------------------------
# Model loading
# ----------------------------

def load_model_and_processor(model_path: str, device):
    """
    Load either a standard CLIP model (HF directory) or a custom vision encoder CLIP checkpoint
    saved by your train_clip_modes.py (we key on substrings in the path).
    """
    # Check for custom model info file first
    custom_info_path = os.path.join(model_path, "custom_model_info.pt")
    if os.path.exists(custom_info_path):
        print(f"Loading custom model using info from {custom_info_path}")
        custom_info = torch.load(custom_info_path, map_location='cpu')
        model_type = custom_info['model_type']
        vision_hidden_size = custom_info.get('vision_hidden_size', 768)
        
        # Base CLIP used to construct the wrapper (for configs)
        base_model_name = "models/clip-vit-base-patch32"
        base_model = CLIPModel.from_pretrained(base_model_name)
        processor = CLIPImageProcessor.from_pretrained(base_model_name)
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create the appropriate model based on type
        if model_type == "resnet18" and ResNet18CLIP is not None:
            print("Loading ResNet18CLIP model from", model_path)
            model = ResNet18CLIP(base_model, vision_hidden_size=vision_hidden_size)
        elif model_type == "densenet121" and DenseNet121CLIP is not None:
            print("Loading DenseNet121CLIP model from", model_path)
            model = DenseNet121CLIP(base_model, vision_hidden_size=vision_hidden_size)
        elif model_type == "vgg11" and VGG11CLIP is not None:
            print("Loading VGG11CLIP model from", model_path)
            model = VGG11CLIP(base_model, vision_hidden_size=vision_hidden_size)
        elif model_type == "small_resnet" and SmallResNetCLIP is not None:
            print("Loading SmallResNetCLIP model from", model_path)
            model = SmallResNetCLIP(base_model, vision_hidden_size=vision_hidden_size)
        elif model_type == "tiny_resnet" and TinyResNetCLIP is not None:
            print("Loading TinyResNetCLIP model from", model_path)
            model = TinyResNetCLIP(base_model, vision_hidden_size=vision_hidden_size)
        else:
            raise ValueError(f"Unknown or unavailable model type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model, processor
    
    # Fallback to legacy path-based detection
    if "tiny" in model_path and "resnet" in model_path and TinyResNetCLIP is not None:
        print("Loading TinyResNetCLIP model from", model_path)
        # Base CLIP used to construct the wrapper (for configs)
        base_model_name = "models/clip-vit-base-patch32"
        base_model = CLIPModel.from_pretrained(base_model_name)
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        vision_hidden_size = checkpoint.get('vision_hidden_size', 256)
        model = TinyResNetCLIP(base_model, vision_hidden_size=vision_hidden_size)
        processor = CLIPImageProcessor.from_pretrained(base_model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model, processor
    elif "resnet18" in model_path and ResNet18CLIP is not None:
        print("Loading ResNet18CLIP model from", model_path)
        base_model_name = "models/clip-vit-base-patch32"
        base_model = CLIPModel.from_pretrained(base_model_name)
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        vision_hidden_size = checkpoint.get('vision_hidden_size', 768)
        model = ResNet18CLIP(base_model, vision_hidden_size=vision_hidden_size)
        processor = CLIPImageProcessor.from_pretrained(base_model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model, processor
    elif "densenet" in model_path and DenseNet121CLIP is not None:
        print("Loading DenseNet121CLIP model from", model_path)
        base_model_name = "models/clip-vit-base-patch32"
        base_model = CLIPModel.from_pretrained(base_model_name)
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        vision_hidden_size = checkpoint.get('vision_hidden_size', 768)
        model = DenseNet121CLIP(base_model, vision_hidden_size=vision_hidden_size)
        processor = CLIPImageProcessor.from_pretrained(base_model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model, processor
    elif "vgg" in model_path and VGG11CLIP is not None:
        print("Loading VGG11CLIP model from", model_path)
        base_model_name = "models/clip-vit-base-patch32"
        base_model = CLIPModel.from_pretrained(base_model_name)
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        vision_hidden_size = checkpoint.get('vision_hidden_size', 768)
        model = VGG11CLIP(base_model, vision_hidden_size=vision_hidden_size)
        processor = CLIPImageProcessor.from_pretrained(base_model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model, processor
    elif "resnet" in model_path and SmallResNetCLIP is not None:
        print("Loading SmallResNetCLIP model from", model_path)
        # Base CLIP used to construct the wrapper (for configs)
        base_model_name = "models/clip-vit-base-patch32"
        base_model = CLIPModel.from_pretrained(base_model_name)
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        vision_hidden_size = checkpoint.get('vision_hidden_size', 512)
        model = SmallResNetCLIP(base_model, vision_hidden_size=vision_hidden_size)
        processor = CLIPImageProcessor.from_pretrained(base_model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model, processor
    else:
        print("Loading CLIP model from", model_path)
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPImageProcessor.from_pretrained(model_path)
        model.to(device).eval()
        return model, processor

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--task", type=str, choices=["heart","triangle","both"], default="both")
    ap.add_argument("--splits", nargs=2, metavar=("TRAIN","EVAL"), default=["train","val"])

    ap.add_argument("--backbone", type=str, choices=["auto","vit","resnet"], default="auto")

    ap.add_argument("--vit_layers", type=int, nargs="+", default=[2,4,6,8,10,12])
    ap.add_argument("--resnet_stages", type=str, nargs="+", default=["layer1","layer2","layer3","layer4"])

    ap.add_argument("--readouts_vit", type=str, nargs="+", default=["cls","mean_tokens","region"])
    ap.add_argument("--readouts_resnet", type=str, nargs="+", default=["gap","gmp","region"])

    ap.add_argument("--probes", type=str, nargs="+", default=["linear","mlp"], help="include 'rff' to enable RFF")
    ap.add_argument("--use_rff", action="store_true")
    ap.add_argument("--rff_dim", type=int, default=4096)
    ap.add_argument("--rff_gamma", type=float, default=0.05)

    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp_dtype", type=str, choices=["fp16","bf16"], default="fp16")

    ap.add_argument("--save_dir", type=str, default="./results/layerwise")
    return ap.parse_args()

# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_dtype = torch.float16 if args.amp_dtype=="fp16" else torch.bfloat16

    model, processor = load_model_and_processor(args.model_path, device)

    # Backbone selection
    bk = args.backbone
    if bk == "auto":
        bk = "vit" if is_vit_clip(model) else "resnet"
    print(f"Backbone detected: {bk.upper()}")

    # Build feature spec
    if bk == "vit":
        spec = FeatureSpec(kind="vit", layers=args.vit_layers, readouts=args.readouts_vit)
    else:
        spec = FeatureSpec(kind="resnet", layers=args.resnet_stages, readouts=args.readouts_resnet)

    # Collect features for train/eval
    train_split, eval_split = args.splits
    feats_tr, y_tr = collect_features(model, processor, args.dataset_id, train_split, args.task,
                                      spec, batch_size=args.batch_size, num_workers=args.num_workers,
                                      device=device, amp_dtype=amp_dtype)
    feats_ev, y_ev = collect_features(model, processor, args.dataset_id, eval_split, args.task,
                                      spec, batch_size=args.batch_size, num_workers=args.num_workers,
                                      device=device, amp_dtype=amp_dtype)

    ensure_dir(args.save_dir)
    model_base = os.path.basename(os.path.normpath(args.model_path))
    out_dir = os.path.join(args.save_dir, f"{model_base}_{bk}_{args.task}")
    ensure_dir(out_dir)

    # Handle both task vs single task
    if args.task == "both":
        # Train probes per (layer, readout) for both tasks
        results = {
            "meta": {
                "dataset": args.dataset_id, "model_path": args.model_path, "task": args.task,
                "backbone": bk, "train_split": train_split, "eval_split": eval_split,
                "probes": args.probes, "seed": args.seed
            }, 
            "scores": {"heart": {}, "triangle": {}}
        }

        for key in sorted(feats_tr.keys()):
            Xtr = feats_tr[key]; Xev = feats_ev[key]
            
            # Heart probes
            heart_results = run_probes(Xtr, y_tr["heart"], Xev, y_ev["heart"], probes=args.probes, device=device,
                                     epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
                                     use_rff=args.use_rff, rff_dim=args.rff_dim, rff_gamma=args.rff_gamma, seed=args.seed)
            results["scores"]["heart"][f"{key[0]}::{key[1]}"] = heart_results
            print(f"[HEART {key[0]} | {key[1]}] -> {heart_results}")
            
            # Triangle probes
            triangle_results = run_probes(Xtr, y_tr["triangle"], Xev, y_ev["triangle"], probes=args.probes, device=device,
                                        epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
                                        use_rff=args.use_rff, rff_dim=args.rff_dim, rff_gamma=args.rff_gamma, seed=args.seed)
            results["scores"]["triangle"][f"{key[0]}::{key[1]}"] = triangle_results
            print(f"[TRIANGLE {key[0]} | {key[1]}] -> {triangle_results}")

        # Save JSON
        with open(os.path.join(out_dir, "layerwise_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {out_dir}/layerwise_results.json")

        
    else:
        # Original single-task logic
        results = {"meta":{
            "dataset": args.dataset_id, "model_path": args.model_path, "task": args.task,
            "backbone": bk, "train_split": train_split, "eval_split": eval_split,
            "probes": args.probes, "seed": args.seed
        }, "scores": {}}

        for key in sorted(feats_tr.keys()):
            Xtr = feats_tr[key]; Xev = feats_ev[key]
            entry={}
            r = run_probes(Xtr, y_tr, Xev, y_ev, probes=args.probes, device=device,
                           epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
                           use_rff=args.use_rff, rff_dim=args.rff_dim, rff_gamma=args.rff_gamma, seed=args.seed)
            entry.update(r)
            results["scores"][f"{key[0]}::{key[1]}"] = entry
            print(f"[{key[0]} | {key[1]}] -> {entry}")

        # Save JSON
        with open(os.path.join(out_dir, "layerwise_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {out_dir}/layerwise_results.json")


if __name__ == "__main__":
    main()
