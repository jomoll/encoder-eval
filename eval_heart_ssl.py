#!/usr/bin/env python3
# eval_heart_triangle.py
import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

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
    p.add_argument("--label_col", default="label")
    p.add_argument("--classes", nargs="+", default=["heart", "triangle"], help="ordered class names")
    p.add_argument("--model", default="vit_small_patch16_224", help="timm backbone name used during pretraining")
    p.add_argument("--ckpt", required=True, help="path to saved backbone state_dict, e.g. epoch_099_dino_backbone.pt")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--probe", choices=["linear", "knn"], default="linear")
    p.add_argument("--probe_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--k", type=int, default=5, help="k for kNN")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="eval_results.json")
    return p.parse_args()

# dataset wrapper
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, image_col, label_col, transform, class_to_idx):
        self.ds = hf_ds
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.class_to_idx = class_to_idx
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex[self.image_col]
        y_raw = ex[self.label_col]
        # allow integer or string labels
        if isinstance(y_raw, str):
            y = self.class_to_idx[y_raw]
        else:
            y = int(y_raw)
        x = self.transform(img)
        return x, y

def build_transform(img_size):
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3,1,1) if x.size(0) == 1 else x),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_backbone(mode, model_name, ckpt_path, device):
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
    return backbone, feat_dim

@torch.no_grad()
def extract_features(backbone, loader, device):
    feats, labels = [], []
    for x, y in loader:
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
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.head(x)

def train_linear_probe(train_feats, train_labels, val_feats, val_labels, in_dim, num_classes, epochs, lr, wd, device):
    model = LinearProbe(in_dim, num_classes).to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    train_feats = train_feats.to(device); train_labels = train_labels.to(device)
    val_feats = val_feats.to(device); val_labels = val_labels.to(device)
    best_acc = 0.0
    for ep in range(1, epochs+1):
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
    # cosine similarity kNN
    train = torch.nn.functional.normalize(train_feats, dim=1)
    test = torch.nn.functional.normalize(test_feats, dim=1)
    sims = test @ train.T  # [Nt, Ntr]
    topk = torch.topk(sims, k=k, dim=1).indices  # [Nt, k]
    preds = []
    for idxs in topk:
        votes = train_labels[idxs].tolist()
        c = Counter(votes).most_common(1)[0][0]
        preds.append(c)
    return torch.tensor(preds, dtype=torch.long)

def accuracy_and_confmat(pred, y, num_classes):
    acc = (pred == y).float().mean().item()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y.tolist(), pred.tolist()):
        conf[t, p] += 1
    return acc, conf

def main():
    args = parse_args()
    device = args.device

    # classes
    class_to_idx = {c:i for i,c in enumerate(args.classes)}
    idx_to_class = {i:c for c,i in class_to_idx.items()}

    # load data
    tfm = build_transform(args.img_size)
    ds_train_hf = load_dataset(args.data, split=args.train_split)
    ds_test_hf  = load_dataset(args.data, split=args.test_split)
    ds_train = HFDataset(ds_train_hf, args.image_col, args.label_col, tfm, class_to_idx)
    ds_test  = HFDataset(ds_test_hf,  args.image_col, args.label_col, tfm, class_to_idx)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # load backbone
    if "mae" in args.ckpt:
        mode = "mae"
    else:
        mode = "dino"
    backbone, feat_dim = load_backbone(mode, args.model, args.ckpt, device)
    print(f"Loaded backbone {args.model} from {args.ckpt}. feat_dim={feat_dim}")

    # features
    with torch.no_grad():
        train_feats, train_labels = extract_features(backbone, dl_train, device)
        test_feats,  test_labels  = extract_features(backbone, dl_test, device)

    results = {"probe": args.probe, "classes": args.classes}

    if args.probe == "linear":
        # split a small validation from train for monitoring
        n = train_feats.size(0)
        perm = torch.randperm(n)
        n_val = max(1, int(0.1 * n))
        val_idx = perm[:n_val]; tr_idx = perm[n_val:]
        lp_model, best_val = train_linear_probe(
            train_feats[tr_idx], train_labels[tr_idx],
            train_feats[val_idx], train_labels[val_idx],
            in_dim=train_feats.size(1), num_classes=len(args.classes),
            epochs=args.probe_epochs, lr=args.lr, wd=args.wd, device=device
        )
        with torch.no_grad():
            lp_model.eval()
            logits = lp_model(test_feats.to(device))
            pred = logits.argmax(1).cpu()
    else:
        pred = knn_predict(train_feats, train_labels, test_feats, k=args.k)

    acc, conf = accuracy_and_confmat(pred, test_labels, num_classes=len(args.classes))
    print(f"Test accuracy: {acc:.4f}")
    print("Confusion matrix (rows true, cols pred):")
    print(conf.numpy())

    # per class accuracy
    per_class = {}
    for i, cname in idx_to_class.items():
        mask = (test_labels == i)
        per_class[cname] = float((pred[mask] == i).float().mean().item()) if mask.any() else float("nan")
    print("Per class accuracy:", per_class)

    # save json
    out = Path(args.out)
    payload = {
        "probe": args.probe,
        "accuracy": acc,
        "per_class_accuracy": per_class,
        "confusion_matrix": conf.tolist(),
        "n_test": int(test_labels.numel()),
        "classes": args.classes,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote results to {out.resolve()}")

if __name__ == "__main__":
    main()
