#!/usr/bin/env python3
# eval_heart_ssl.py
import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from collections import Counter

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/silent-heart-dataset", help="HF dataset name or local path")
    p.add_argument("--train_split", default="train")
    p.add_argument("--test_split", default="test")
    p.add_argument("--image_col", default="image")
    p.add_argument("--label_col", default="H")
    p.add_argument("--task", choices=["heart", "triangle", "both"], default="both", help="Task to evaluate")
    p.add_argument("--model", default="vit_base_patch16_224", help="timm backbone name used during pretraining")
    p.add_argument("--ckpt", required=True, help="path to saved backbone state_dict, e.g. epoch_099_dino_backbone.pt")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--probe", choices=["linear", "knn"], default="linear")
    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.001)
    p.add_argument("--k", type=int, default=5, help="k for kNN")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="")
    return p.parse_args()

def extract_named_list(ex):
    """
    Try multiple fields to get named objects metadata.
    Returns a list of dicts with keys including 'shape'.
    """
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

def has_triangle_named(ex):
    try:
        named = extract_named_list(ex)
        for o in named:
            shp = str(o.get("shape", "")).lower()
            if shp == "triangle":
                return 1
        return 0
    except Exception:
        return 0

def extract_heart_label(ex):
    """Extract heart label using the same robust logic as eval_heart_probe.py"""
    if "H" in ex:
        v = ex["H"]
        if isinstance(v, int):
            return v
        else:
            try:
                if isinstance(v, str):
                    return 1 if v.lower().strip() in ["1","heart","true","yes"] else 0
                else:
                    return int(v)
            except Exception:
                return 0
    elif "id" in ex and isinstance(ex["id"], str) and ex["id"][-2:] in ["_0","_1"]:
        return int(ex["id"][-1])
    elif "metadata" in ex and isinstance(ex["metadata"], dict) and "H" in ex["metadata"]:
        return int(ex["metadata"]["H"])
    else:
        raise ValueError("Could not resolve label H for example; ensure dataset has 'H' or id ends with _0/_1.")

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, image_col, transform, task="both"):
        self.ds = hf_ds
        self.image_col = image_col
        self.transform = transform
        self.task = task
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex[self.image_col]
        x = self.transform(img)
        
        if self.task == "both":
            # Return both labels
            heart_label = extract_heart_label(ex)
            triangle_label = has_triangle_named(ex)
            return x, heart_label, triangle_label
        elif self.task == "heart":
            y = extract_heart_label(ex)
            return x, y
        elif self.task == "triangle":
            y = has_triangle_named(ex)
            return x, y

def build_transform(img_size, mode):
    if mode == "mae":
        # MAE expects [0,1] range - no normalization
        return T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3,1,1) if x.size(0) == 1 else x),
            # No normalization for MAE
        ])
    else:  # DINO
        # DINO uses ImageNet normalization
        return T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3,1,1) if x.size(0) == 1 else x),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

def load_backbone(mode, model_name, ckpt_path, device):
    if model_name is None:
        model_name = "vit_base_patch16_224" if mode == "mae" else "vit_small_patch16_224"
    
    if mode == "mae":
        from train_ssl import MAEWrapper
        model = MAEWrapper(model_name=model_name, img_size=224)
        backbone = model.backbone()
    else:  # dino
        import timm
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    
    sd = torch.load(ckpt_path, map_location="cpu")
    backbone.load_state_dict(sd, strict=True)
    backbone.eval().to(device)
    
    feat_dim = getattr(backbone, "num_features", None)
    if feat_dim is None:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = backbone(dummy_input)
            feat_dim = dummy_output.shape[-1]
    
    return backbone, feat_dim

@torch.no_grad()
def extract_features(backbone, loader, device, task="both"):
    feats = []
    if task == "both":
        heart_labels, triangle_labels = [], []
        for batch in tqdm(loader, desc="Extracting features"):
            x, h_labels, t_labels = batch
            x = x.to(device, non_blocking=True)
            f = backbone(x)
            if f.ndim > 2:
                f = f.flatten(1)
            feats.append(f.cpu())
            heart_labels.append(h_labels)
            triangle_labels.append(t_labels)
        feats = torch.cat(feats, dim=0).contiguous()
        heart_labels = torch.cat(heart_labels, dim=0).contiguous()
        triangle_labels = torch.cat(triangle_labels, dim=0).contiguous()
        return feats, heart_labels, triangle_labels
    else:
        labels = []
        for x, y in tqdm(loader, desc="Extracting features"):
            x = x.to(device, non_blocking=True)
            f = backbone(x)
            if f.ndim > 2:
                f = f.flatten(1)
            feats.append(f.cpu())
            labels.append(y)
        feats = torch.cat(feats, dim=0).contiguous()
        labels = torch.cat(labels, dim=0).contiguous()
        return feats, labels

class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.head(x)

def train_linear_probe(train_feats, train_labels, val_feats, val_labels, in_dim, epochs, lr, wd, device):
    model = LinearProbe(in_dim, num_classes=2).to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    train_feats = train_feats.to(device); train_labels = train_labels.to(device)
    val_feats = val_feats.to(device); val_labels = val_labels.to(device)
    
    best_acc = 0.0
    for ep in tqdm(range(1, epochs+1), desc="Training Linear Probe"):
        model.train()
        logits = model(train_feats)
        loss = criterion(logits, train_labels)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        
        with torch.no_grad():
            model.eval()
            pred = model(val_feats).argmax(1)
            acc = (pred == val_labels).float().mean().item()
        print(f"[probe] epoch {ep:02d} loss {loss.item():.4f} acc {acc:.4f}")
        best_acc = max(best_acc, acc)
    
    return model, best_acc

def knn_predict(train_feats, train_labels, test_feats, k):
    train = torch.nn.functional.normalize(train_feats, dim=1)
    test = torch.nn.functional.normalize(test_feats, dim=1)
    sims = test @ train.T
    topk = torch.topk(sims, k=k, dim=1).indices
    preds = []
    for idxs in topk:
        votes = train_labels[idxs].tolist()
        c = Counter(votes).most_common(1)[0][0]
        preds.append(c)
    return torch.tensor(preds, dtype=torch.long)

def accuracy_and_confmat(pred, y, num_classes=2):
    acc = (pred == y).float().mean().item()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y.tolist(), pred.tolist()):
        conf[t, p] += 1
    return acc, conf

def evaluate_single_task(args, backbone, feat_dim, device, task_name):
    """Evaluate a single task (heart or triangle)"""
    print(f"\n=== Evaluating {task_name} task ===")
    
    # Load data for single task
    tfm = build_transform(args.img_size, args.mode)
    ds_train_hf = load_dataset(args.data, split=args.train_split)
    ds_test_hf = load_dataset(args.data, split=args.test_split)
    ds_train = HFDataset(ds_train_hf, args.image_col, tfm, task=task_name)
    ds_test = HFDataset(ds_test_hf, args.image_col, tfm, task=task_name)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # Extract features
    with torch.no_grad():
        train_feats, train_labels = extract_features(backbone, dl_train, device, task=task_name)
        test_feats, test_labels = extract_features(backbone, dl_test, device, task=task_name)
    
    print(f"Extracted features: train {train_feats.shape}, test {test_feats.shape}")
    
    # Train probe and evaluate
    if args.probe == "linear":
        n = train_feats.size(0)
        perm = torch.randperm(n)
        n_val = max(1, int(0.1 * n))
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        
        model, best_val = train_linear_probe(
            train_feats[tr_idx], train_labels[tr_idx],
            train_feats[val_idx], train_labels[val_idx],
            in_dim=feat_dim, epochs=args.probe_epochs, lr=args.lr, wd=args.wd, device=device
        )
        
        with torch.no_grad():
            model.eval()
            logits = model(test_feats.to(device))
            pred = logits.argmax(1).cpu()
    else:
        pred = knn_predict(train_feats, train_labels, test_feats, k=args.k)
    
    acc, conf = accuracy_and_confmat(pred, test_labels, num_classes=2)
    
    # Per class accuracy
    classes = [f"no_{task_name}", task_name]
    per_class = {}
    for i, cname in enumerate(classes):
        mask = (test_labels == i)
        per_class[cname] = float((pred[mask] == i).float().mean().item()) if mask.any() else float("nan")
    
    print(f"{task_name} accuracy: {acc:.4f}")
    print("Confusion matrix (rows true, cols pred):")
    print(conf.numpy())
    print("Per class accuracy:", per_class)
    
    return {
        "task": task_name,
        "probe": args.probe,
        "accuracy": acc,
        "per_class_accuracy": per_class,
        "confusion_matrix": conf.tolist(),
        "n_test": int(test_labels.numel()),
        "classes": classes,
    }

def main():
    args = parse_args()
    device = args.device
    
    # Load backbone
    if "mae" in args.ckpt:
        args.mode = "mae"
    else:
        args.mode = "dino"
    backbone, feat_dim = load_backbone(args.mode, args.model, args.ckpt, device)
    print(f"Loaded backbone {args.model} from {args.ckpt}. feat_dim={feat_dim}")
    
    results = {}
    
    if args.task == "both":
        # Extract features once for both tasks
        print("\n=== Extracting features for both tasks ===")
        tfm = build_transform(args.img_size, args.mode)
        ds_train_hf = load_dataset(args.data, split=args.train_split)
        ds_test_hf = load_dataset(args.data, split=args.test_split)
        ds_train = HFDataset(ds_train_hf, args.image_col, tfm, task="both")
        ds_test = HFDataset(ds_test_hf, args.image_col, tfm, task="both")
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        
        with torch.no_grad():
            train_feats, train_heart_labels, train_triangle_labels = extract_features(backbone, dl_train, device, task="both")
            test_feats, test_heart_labels, test_triangle_labels = extract_features(backbone, dl_test, device, task="both")
        
        print(f"Extracted features: train {train_feats.shape}, test {test_feats.shape}")
        
        # Evaluate heart task
        heart_results = evaluate_task_with_features(
            args, train_feats, train_heart_labels, test_feats, test_heart_labels, 
            feat_dim, device, "heart"
        )
        
        # Evaluate triangle task
        triangle_results = evaluate_task_with_features(
            args, train_feats, train_triangle_labels, test_feats, test_triangle_labels,
            feat_dim, device, "triangle"
        )
        
        results = {
            "heart": heart_results,
            "triangle": triangle_results,
            "summary": {
                "heart_accuracy": heart_results["accuracy"],
                "triangle_accuracy": triangle_results["accuracy"],
            }
        }
        
        print(f"\n=== SUMMARY ===")
        print(f"Heart accuracy: {heart_results['accuracy']:.4f}")
        print(f"Triangle accuracy: {triangle_results['accuracy']:.4f}")
        
    else:
        # Evaluate single task
        results = evaluate_single_task(args, backbone, feat_dim, device, args.task)
    
    # Save results
    out = Path("outputs/ssl/" + args.out + "/all_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {out.resolve()}")

def evaluate_task_with_features(args, train_feats, train_labels, test_feats, test_labels, feat_dim, device, task_name):
    """Evaluate a single task using pre-computed features"""
    print(f"\n=== Evaluating {task_name} task ===")
    
    # Train probe and evaluate
    if args.probe == "linear":
        n = train_feats.size(0)
        perm = torch.randperm(n)
        n_val = max(1, int(0.1 * n))
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        
        model, best_val = train_linear_probe(
            train_feats[tr_idx], train_labels[tr_idx],
            train_feats[val_idx], train_labels[val_idx],
            in_dim=feat_dim, epochs=args.probe_epochs, lr=args.lr, wd=args.wd, device=device
        )
        
        with torch.no_grad():
            model.eval()
            logits = model(test_feats.to(device))
            pred = logits.argmax(1).cpu()
    else:
        pred = knn_predict(train_feats, train_labels, test_feats, k=args.k)
    
    acc, conf = accuracy_and_confmat(pred, test_labels, num_classes=2)
    
    # Per class accuracy
    classes = [f"no_{task_name}", task_name]
    per_class = {}
    for i, cname in enumerate(classes):
        mask = (test_labels == i)
        per_class[cname] = float((pred[mask] == i).float().mean().item()) if mask.any() else float("nan")
    
    print(f"{task_name} accuracy: {acc:.4f}")
    print("Confusion matrix (rows true, cols pred):")
    print(conf.numpy())
    print("Per class accuracy:", per_class)
    
    return {
        "task": task_name,
        "probe": args.probe,
        "accuracy": acc,
        "per_class_accuracy": per_class,
        "confusion_matrix": conf.tolist(),
        "n_test": int(test_labels.numel()),
        "classes": classes,
    }
if __name__ == "__main__":
    main()