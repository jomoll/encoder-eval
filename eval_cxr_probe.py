#!/usr/bin/env python3
"""
Evaluate presence of a silent or named object in CLIP image embeddings
======================================================================

This script loads a (fine-tuned) CLIP-like model, freezes it, encodes images,
and trains a small linear classifier on the frozen image embeddings to predict
object presence.

Supported tasks
---------------
- marker    : predicts laterality marker (L=1, R=0)
- pleural_effusion : predicts pleural effusion presence (0/1)

Usage
-----
pip install torch torchvision transformers datasets pillow tqdm scikit-learn

# Example: marker task
python eval_heart_probe.py \
  --dataset_id jomoll/silent-heart-dataset \
  --model_path ./ckpt_standard/best \
  --task marker \
  --split_train train --split_eval val \
  --tune_threshold_on eval --tune_policy acc --calibrate_on train

# Example: pleural_effusion task
python eval_heart_probe.py \
  --dataset_id jomoll/silent-heart-dataset \
  --model_path ./ckpt_standard/best \
  --task pleural_effusion \
  --split_train train --split_eval val \
  --tune_threshold_on eval --tune_policy acc
"""

import os, argparse, json, math, random
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

from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, roc_curve

# Optional custom vision backbone wrapper from your training script
try:
    from train_clip_modes import SmallResNetCLIP, TinyResNetCLIP, ResNet18CLIP, DenseNet121CLIP, VGG11CLIP  # noqa: F401
except Exception:
    SmallResNetCLIP = None
    TinyResNetCLIP = None
    ResNet18CLIP = None
    DenseNet121CLIP = None
    VGG11CLIP = None

# ----------------------------
# Model loading
# ----------------------------

def load_model(model_path: str, device):
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
        return model
    
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        return model
    else:
        print("Loading CLIP model from", model_path)
        model = CLIPModel.from_pretrained(model_path)
        model.to(device).eval()
        return model

# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Transforms (eval-style)
# ----------------------------

class EvalTransform:
    def __init__(self, image_size: int, mean, std):
        self.resize = T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC)
        self.center = T.CenterCrop(image_size)
        self.totensor = T.ToTensor()
        self.norm = T.Normalize(mean=mean, std=std)
    def __call__(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.resize(img)
        img = self.center(img)
        return self.norm(self.totensor(img))

def mask_special_region(img: Image.Image, ex: Dict[str, Any]) -> Image.Image:
    """
    Black out a circular region around the special object's center (heart-only audit).
    Expects 'special_object' JSON with cx, cy in HR coords and 'HR' scalar in the example.
    """
    try:
        if "special_object" not in ex:
            return img
        so = ex["special_object"]
        if isinstance(so, str):
            so = json.loads(so)
        cx_hr, cy_hr = float(so["cx"]), float(so["cy"])
        HR = float(ex.get("HR", 448))
        W, H = img.size
        # map to current image coords
        cx = cx_hr / HR * W
        cy = cy_hr / HR * H
        r = float(so.get("size", 22.0)) / HR * W * 1.4  # slightly larger than object
        # draw black circle
        masked = img.copy()
        d = ImageDraw.Draw(masked)
        bbox = (cx - r, cy - r, cx + r, cy + r)
        d.ellipse(bbox, fill=(0,0,0))
        return masked
    except Exception:
        return img


# ----------------------------
# Collator
# ----------------------------

def make_collate_fn(task: str):
    def collate(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        labels = [b["label"] for b in batch]
        return {"pixel_values": pixel_values, "label": labels}
    return collate

# ----------------------------
# Dataset wrapper with with_transform
# ----------------------------

def extract_named_list(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Try multiple fields to get named objects metadata.
    Returns a list of dicts with keys including 'shape'.
    """
    # direct fields that might exist
    candidates = ["named_objects", "named", "named_captions"]
    for key in candidates:
        if key in ex and ex[key] is not None:
            val = ex[key]
            try:
                if isinstance(val, str):
                    return json.loads(val)
                if isinstance(val, list):
                    return val
            except Exception:
                pass
    # nested in 'metadata'
    if "metadata" in ex and ex["metadata"] is not None:
        m = ex["metadata"]
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                m = None
        if isinstance(m, dict):
            for key in ["named_objects", "named", "named_captions"]:
                if key in m and m[key] is not None:
                    val = m[key]
                    try:
                        if isinstance(val, str):
                            return json.loads(val)
                        if isinstance(val, list):
                            return val
                    except Exception:
                        pass
    return []

def has_triangle_named(ex: Dict[str, Any]) -> int:
    try:
        named = extract_named_list(ex)
        for o in named:
            shp = str(o.get("shape", "")).lower()
            if shp == "triangle":
                return 1
        return 0
    except Exception:
        return 0

def build_dataset_with_labels(dataset_id: str,
                              image_processor: CLIPImageProcessor,
                              split: str,
                              task: str = "marker",
                              mask_special: bool = False):
    ds = load_dataset(dataset_id, split=split)
    
    # Filter out invalid samples
    def is_valid_sample(example):
        # Invalid if marker_present is False or pleural_effusion_confidence is 0
        marker_present = example.get("marker_present", True)  # default True if missing
        pleural_effusion_confidence = example.get("pleural_effusion_confidence", 1)  # default 1 if missing
        
        return marker_present != False and pleural_effusion_confidence != 0
    
    # Count samples before filtering
    original_count = len(ds)
    
    # Apply filter
    ds = ds.filter(is_valid_sample)
    
    # Count samples after filtering
    filtered_count = len(ds)
    invalid_count = original_count - filtered_count
    
    print(f"Dataset filtering for {split}:")
    print(f"  Original samples: {original_count}")
    print(f"  Valid samples: {filtered_count}")
    print(f"  Invalid samples discarded: {invalid_count}")
    
    image_size = image_processor.size["shortest_edge"]
    tf = EvalTransform(image_size, image_processor.image_mean, image_processor.image_std)

    # helper: resolve labels based on task
    def resolve_labels(ex):
        if task == "marker":
            if "marker" in ex:
                # Convert L to 1, R (or anything else) to 0
                return 1 if ex["marker"] == "L" else 0
            else:
                raise ValueError("Could not resolve 'marker' field for laterality task")
        elif task == "pleural_effusion":
            if "pleural_effusion_present" in ex:
                return int(ex["pleural_effusion_present"])
            else:
                raise ValueError("Could not resolve 'pleural_effusion_present' field for pleural effusion task")
        else:
            raise ValueError(f"Unknown task: {task}")

    def _transform(examples):
        # Handle both single example and batched examples
        is_batched = isinstance(examples["image"], list)
        if is_batched:
            images = examples["image"]
            pixel_values, labels = [], []
            for i, img in enumerate(images):
                ex_single = {k: (v[i] if isinstance(v, list) else v) for k, v in examples.items()}
                if mask_special:  # Remove heart-specific condition
                    img = mask_special_region(img, ex_single)
                pixel_values.append(tf(img))
                labels.append(resolve_labels(ex_single))
            examples["pixel_values"] = pixel_values
            examples["label"] = labels
        else:
            img = examples["image"]
            if mask_special:  # Remove heart-specific condition
                img = mask_special_region(img, examples)
            examples["pixel_values"] = tf(img)
            examples["label"] = resolve_labels(examples)
        return examples

    ds = ds.with_transform(_transform)
    return ds

# ----------------------------
# Probe (linear classifier)
# ----------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)
    def forward(self, x):
        return self.fc(x)

# ----------------------------
# Embedding
# ----------------------------

def get_image_features_any(model, pixel_values):
    """
    Support both HF CLIPModel and custom wrappers that implement get_image_features.
    """
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=pixel_values)
    # Fallback: try forward and pick image_embeds
    out = model(pixel_values=pixel_values, return_loss=False)
    if hasattr(out, "image_embeds"):
        return out.image_embeds
    raise AttributeError("Model does not provide get_image_features or image_embeds.")

def embed_split(model, loader: DataLoader, device, amp_dtype, task: str):
    model.eval()
    feats = []
    labels = []
        
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Embedding {task}"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            y = torch.tensor(batch["label"], dtype=torch.long)
                
            with autocast(dtype=amp_dtype):
                emb = get_image_features_any(model, pixel_values)
            feats.append(emb.detach().cpu())
            labels.append(y)
                
    X = torch.cat(feats, dim=0)
    y = torch.cat(labels, dim=0)
    return X, y

# ----------------------------
# Probe training
# ----------------------------

def train_probe(Xtr, ytr, epochs=50, lr=5e-3, wd=0.0, device="cpu"):
    in_dim = Xtr.shape[1]
    probe = LinearProbe(in_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    probe.train()
    for e in range(epochs):
        bs = min(4096, Xtr.size(0))
        perm = torch.randperm(Xtr.size(0), device=device)
        for i in range(0, Xtr.size(0), bs):
            idx = perm[i:i+bs]
            logits = probe(Xtr[idx])
            loss = ce(logits, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    probe.eval()
    return probe

# ----------------------------
# Threshold tuning & calibration
# ----------------------------

def pick_threshold(y_true: np.ndarray, scores: np.ndarray, policy: str = "acc"):
    fpr, tpr, thr = roc_curve(y_true, scores)
    best_thr = 0.5
    best_metric = -1.0
    for t in thr:
        pred = (scores >= t).astype(int)
        if policy == "acc":
            m = accuracy_score(y_true, pred)
        elif policy == "youden":
            i = np.argmin(np.abs(thr - t))
            m = tpr[i] - fpr[i]
        elif policy == "f1":
            m = f1_score(y_true, pred)
        else:
            m = accuracy_score(y_true, pred)
        if m > best_metric:
            best_metric = m
            best_thr = float(t)
    return best_thr, best_metric

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros([]))  # T=1 initial
    def forward(self, logits):
        T = torch.exp(self.log_T)
        return logits / T

def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter=200, lr=0.05):
    scaler = TemperatureScaler().to(logits.device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = ce(scaler(logits), labels)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        T = torch.exp(scaler.log_T).item()
    return scaler, T

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=str, default="data/mimic-cxr-combined-annotations")
    ap.add_argument("--model_path", type=str, required=True, help="Path or HF id for the fine-tuned model")
    ap.add_argument("--task", type=str, choices=["marker", "pleural_effusion", "both"], default="both")
    ap.add_argument("--split_train", type=str, default="train_reports")
    ap.add_argument("--split_eval", type=str, default="train_reports")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp_dtype", type=str, choices=["fp16","bf16"], default="fp16")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--mask_special", action="store_true", help="Mask out special regions (if applicable)")
    ap.add_argument("--tune_threshold_on", type=str, choices=["train","eval","none"], default="eval")
    ap.add_argument("--tune_policy", type=str, choices=["acc","youden","f1"], default="acc")
    ap.add_argument("--calibrate_on", type=str, choices=["none","train","eval"], default="none")
    ap.add_argument("--save_dir", type=str, default="./results/probe_out")
    args = ap.parse_args()

    # Build save directory: include model basename and task
    model_base = os.path.basename(os.path.normpath(args.model_path))
    mode_save_dir = os.path.join(args.save_dir, f"{model_base}_task-{args.task}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    # Load model + processor
    model = load_model(args.model_path, device)
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        processor = CLIPImageProcessor.from_pretrained(args.model_path)
    except Exception:
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Determine which tasks to run
    tasks_to_run = []
    if args.task == "both":
        tasks_to_run = ["marker", "pleural_effusion"]
    else:
        tasks_to_run = [args.task]

    all_results = {}
    
    for task in tasks_to_run:
        print(f"\n=== Running task: {task} ===")
        
        # Build datasets and loaders for this task
        ds_train = build_dataset_with_labels(args.dataset_id, processor, split=args.split_train, task=task, mask_special=args.mask_special)
        ds_eval  = build_dataset_with_labels(args.dataset_id, processor, split=args.split_eval,  task=task, mask_special=args.mask_special)

        collate = make_collate_fn(task)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type=="cuda"), collate_fn=collate)
        dl_eval  = DataLoader(ds_eval,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type=="cuda"), collate_fn=collate)

        # Embed
        Xtr, ytr = embed_split(model, dl_train, device, amp_dtype, task)
        Xev, yev = embed_split(model, dl_eval, device, amp_dtype, task)
        
        # Convert embeddings to float32
        Xtr = Xtr.float().to(device); ytr = ytr.to(device)
        Xev = Xev.float().to(device); yev = yev.to(device)

        # Train linear probe
        probe = train_probe(Xtr, ytr, epochs=args.epochs, lr=args.lr, wd=args.weight_decay, device=device)

        # Collect logits on splits
        probe.eval()
        with torch.no_grad():
            logits_tr = probe(Xtr).cpu()
            logits_ev = probe(Xev).cpu()
        p_tr = logits_tr.softmax(dim=1)[:,1].numpy()
        p_ev = logits_ev.softmax(dim=1)[:,1].numpy()

        # Calibration (optional)
        T_fit = 1.0
        if args.calibrate_on != "none":
            if args.calibrate_on == "train":
                scaler, T_fit = calibrate_temperature(logits_tr.to(device), ytr.to(device))
            else:
                scaler, T_fit = calibrate_temperature(logits_ev.to(device), yev.to(device))
            with torch.no_grad():
                logits_tr = scaler(logits_tr.to(device)).cpu()
                logits_ev = scaler(logits_ev.to(device)).cpu()
            p_tr = logits_tr.softmax(dim=1)[:,1].numpy()
            p_ev = logits_ev.softmax(dim=1)[:,1].numpy()

        # Threshold tuning
        best_thr = 0.5
        best_ref = None
        if args.tune_threshold_on == "train":
            best_thr, _ = pick_threshold(ytr.cpu().numpy(), p_tr, policy=args.tune_policy)
            best_ref = "train"
        elif args.tune_threshold_on == "eval":
            best_thr, _ = pick_threshold(yev.cpu().numpy(), p_ev, policy=args.tune_policy)
            best_ref = "eval"

        # Metrics (eval split)
        auc = roc_auc_score(yev.cpu().numpy(), p_ev)
        pred_default = (p_ev >= 0.5).astype(int)
        pred_argmax  = logits_ev.argmax(dim=1).numpy()
        pred_best    = (p_ev >= best_thr).astype(int)

        acc_default = accuracy_score(yev.cpu().numpy(), pred_default)
        acc_argmax  = accuracy_score(yev.cpu().numpy(), pred_argmax)
        acc_best    = accuracy_score(yev.cpu().numpy(), pred_best)
        bal_acc     = balanced_accuracy_score(yev.cpu().numpy(), pred_best)
        f1          = f1_score(yev.cpu().numpy(), pred_best)

        # Store results for this task
        task_results = {
            "task": task,
            "eval_auc": float(auc),
            "eval_acc_default@0.5": float(acc_default),
            "eval_acc_argmax": float(acc_argmax),
            "eval_acc_best": float(acc_best),
            "best_threshold": float(best_thr),
            "threshold_ref": best_ref,
            "balanced_accuracy": float(bal_acc),
            "f1": float(f1),
            "temperature": float(T_fit),
            "n_train": int(len(ds_train)),
            "n_eval": int(len(ds_eval))
        }
        
        all_results[task] = task_results
        
        # Save task-specific results
        task_save_dir = os.path.join(mode_save_dir, task)
        os.makedirs(task_save_dir, exist_ok=True)
        
        with open(os.path.join(task_save_dir, "metrics.json"), "w") as f:
            json.dump(task_results, f, indent=2)
        """
        # Save artifacts for this task
        torch.save({"X_train": Xtr.cpu(), "y_train": ytr.cpu(), "X_eval": Xev.cpu(), "y_eval": yev.cpu()},
                   os.path.join(task_save_dir, "embeddings.pt"))
        torch.save(probe.state_dict(), os.path.join(task_save_dir, "linear_probe.pt"))
        """
        print(f"Results for {task}:")
        print(json.dumps(task_results, indent=2))
        print(f"Saved {task} probe artifacts to {task_save_dir}")

    # Save combined results if running both tasks
    if len(tasks_to_run) > 1:
        os.makedirs(mode_save_dir, exist_ok=True)
        with open(os.path.join(mode_save_dir, "all_metrics.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved combined results to {mode_save_dir}/all_metrics.json")

if __name__ == "__main__":
    main()