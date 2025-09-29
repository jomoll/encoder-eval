#!/usr/bin/env python3
# train_ssl_hf.py
"""
MAE and DINO training on a Hugging Face dataset with columns:
  image: PIL.Image or numpy array
  captions: string (unused here)

Examples
  python train_ssl_hf.py --mode mae  --data jomoll/TAIX-reasoning-v2.1 --split train --out runs/mae
  python train_ssl_hf.py --mode dino --data jomoll/TAIX-reasoning-v2.1 --split train --out runs/dino
"""

import argparse, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset

import torchvision.transforms as T

try:
    import timm
except Exception as e:
    raise RuntimeError("Install timm") from e

# DINO parts
try:
    from lightly.data.multi_view_collate import MultiViewCollate
    from lightly.transforms.utils import IMAGENET_NORMALIZE
    from lightly.loss import DINOLoss as LightlyDINOLoss
    _has_lightly = True
except Exception:
    _has_lightly = False


# ------------------ args ------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mae", "dino"], required=True)
    p.add_argument("--data", type=str, required=True, help="datasets.load_dataset name or path")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--mask_ratio", type=float, default=0.75, help="MAE")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image_col", type=str, default="image")
    p.add_argument("--caption_col", type=str, default="captions")
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)


# ------------------ dataset wrappers ------------------
class HFViewsDataset(torch.utils.data.Dataset):
    """
    Wraps a HF dataset and applies a transform that returns either a single tensor (MAE)
    or a list of tensors (DINO multi-crop). Captions are ignored.
    """
    def __init__(self, hf_ds, image_col, transform):
        self.ds = hf_ds
        self.image_col = image_col
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex[self.image_col]
        return self.transform(img), 0  # dummy label for DataLoader API


def build_mae_transform(img_size):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # stays in [0,1] as MAE expects
    ])


class MultiCropTransform:
    def __init__(self, img_size, n_global=2, n_local=6):
        self.n_global, self.n_local = n_global, n_local
        self.global_t = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(**IMAGENET_NORMALIZE),
        ])
        self.local_t = T.Compose([
            T.RandomResizedCrop(96, scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(**IMAGENET_NORMALIZE),
        ])

    def __call__(self, img):
        crops = [self.global_t(img) for _ in range(self.n_global)]
        crops += [self.local_t(img) for _ in range(self.n_local)]
        return crops


# ------------------ MAE ------------------
class MAEWrapper(nn.Module):
    def __init__(self, model_name="mae_vit_base_patch16", mask_ratio=0.75, img_size=224):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, img_size=img_size, mask_ratio=mask_ratio)
    def forward(self, x):
        return self.model(x)  # returns (loss, pred, mask)
    def backbone(self):
        return self.model.encoder


# ------------------ DINO ------------------
def build_dino_student_teacher(model_name="vit_small_patch16_224"):
    # backbone that returns features
    student_backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    teacher_backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    feat_dim = student_backbone.num_features
    head = nn.Sequential(nn.Linear(feat_dim, 2048), nn.GELU(), nn.Linear(2048, 256))
    s = nn.Sequential(student_backbone, nn.Flatten(1), head)
    t = nn.Sequential(teacher_backbone, nn.Flatten(1), nn.Sequential(nn.Linear(feat_dim, 2048), nn.GELU(), nn.Linear(2048, 256)))
    for tp, sp in zip(t.parameters(), s.parameters()):
        tp.data.copy_(sp.data); tp.requires_grad_(False)
    return s, t, student_backbone


@torch.no_grad()
def update_ema(teacher, student, m):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data = tp.data * m + sp.data * (1 - m)


# ------------------ utils ------------------
def cosine_lr(optimizer, base_lr, epoch, max_epochs):
    for pg in optimizer.param_groups:
        pg['lr'] = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))


def save_backbone(state_dir, epoch, mode, backbone):
    path = Path(state_dir) / f"epoch_{epoch:03d}_{mode}_backbone.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state_dict(), path)
    return str(path)


# ------------------ main ------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # load HF dataset
    hf_ds = load_dataset(args.data, split=args.split)

    if args.mode == "mae":
        transform = build_mae_transform(args.img_size)
        ds = HFViewsDataset(hf_ds, args.image_col, transform)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)

        model_name = args.model or "mae_vit_base_patch16"
        model = MAEWrapper(model_name=model_name, mask_ratio=args.mask_ratio, img_size=args.img_size).to(device)
        lr = args.lr or 1.5e-4
        opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args.wd)

        for epoch in range(1, args.epochs + 1):
            model.train(); cosine_lr(opt, lr, epoch-1, args.epochs)
            loss_sum, n = 0.0, 0
            for imgs, _ in dl:
                imgs = imgs.to(device, non_blocking=True)
                loss, _, _ = model(imgs)
                (loss / args.accum).backward()
                if (n + 1) % args.accum == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
                loss_sum += loss.item(); n += 1
            ckpt = save_backbone(out, epoch, "mae", model.backbone())
            print(f"[{epoch:03d}] loss={loss_sum/n:.4f} saved={ckpt}")

    else:
        if not _has_lightly:
            raise RuntimeError("Install lightly for DINO")
        transform = MultiCropTransform(args.img_size)
        ds = HFViewsDataset(hf_ds, args.image_col, transform)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, collate_fn=MultiViewCollate())

        model_name = args.model or "vit_small_patch16_224"
        student, teacher, backbone = build_dino_student_teacher(model_name)
        student, teacher = student.to(device), teacher.to(device)
        dino_loss = LightlyDINOLoss(out_dim=256, nepochs=args.epochs).to(device)

        lr = args.lr or 5e-4
        opt = optim.AdamW(student.parameters(), lr=lr, weight_decay=args.wd)

        for epoch in range(1, args.epochs + 1):
            student.train(); teacher.eval()
            cosine_lr(opt, lr, epoch-1, args.epochs)
            loss_sum, n = 0.0, 0
            # cosine EMA schedule
            m = 0.996 - 0.3 * (1 + math.cos(math.pi * (epoch-1) / args.epochs)) / 2.0
            for views, _ in dl:
                views = [v.to(device, non_blocking=True) for v in views]
                s_outs = [student(v) for v in views]
                with torch.no_grad():
                    t_outs = [teacher(v) for v in views[:2]]
                loss = dino_loss(s_outs, t_outs)
                (loss / args.accum).backward()
                if (n + 1) % args.accum == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
                loss_sum += loss.item(); n += 1
            update_ema(teacher, student, m)
            ckpt = save_backbone(out, epoch, "dino", backbone.to(device))
            print(f"[{epoch:03d}] loss={loss_sum/n:.4f} saved={ckpt}")


if __name__ == "__main__":
    main()
