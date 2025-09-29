#!/usr/bin/env python3
# train_ssl.py
"""
Minimal MAE/DINO trainers with ViT backbones.
Usage:
  python train_ssl.py --mode mae --data /path/to/images --out runs/mae
  python train_ssl.py --mode dino --data /path/to/images --out runs/dino \
      --epochs 100 --batch-size 128 --lr 5e-4
"""

import argparse, math, os, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as T
import torchvision.datasets as dsets

# ---- optional deps ----
try:
    import timm  # MAE backbone lives here
except Exception as e:
    raise RuntimeError("This script requires `timm` (pip install timm).") from e

try:
    import lightly
    from lightly.data.multi_view_collate import MultiViewCollate
    from lightly.transforms.utils import IMAGENET_NORMALIZE
except Exception:
    lightly = None  # only needed for DINO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mae", "dino"], required=True)
    p.add_argument("--data", type=str, required=True, help="ImageFolder root")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=None, help="If None, set by mode")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", type=str, default=None,
                   help="timm model name. Defaults: mae_vit_base_patch16, vit_small_patch16_224")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--mask-ratio", type=float, default=0.75, help="MAE only")
    p.add_argument("--accum", type=int, default=1, help="grad accumulation steps")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# --------------------- Data ---------------------
def build_dataset(args):
    if args.mode == "mae":
        # Standard strong crops as in MAE (no heavy color jitter is fine)
        tfm = T.Compose([
            T.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # MAE operates on normalized pixels in [0,1], normalization is done inside loss
        ])
        ds = dsets.ImageFolder(args.data, transform=tfm)
        collate_fn = None
    else:
        if lightly is None:
            raise RuntimeError("DINO mode requires `lightly` (pip install lightly).")
        # Multi-crop setup similar to DINO (2 global + 6 local)
        global_crop = T.Compose([
            T.RandomResizedCrop(args.img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(**IMAGENET_NORMALIZE),
        ])
        local_crop = T.Compose([
            T.RandomResizedCrop(96, scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(**IMAGENET_NORMALIZE),
        ])

        class MultiCropTransform:
            def __init__(self, g_tfm, l_tfm, n_global=2, n_local=6):
                self.g_tfm, self.l_tfm = g_tfm, l_tfm
                self.n_global, self.n_local = n_global, n_local
            def __call__(self, x):
                crops = [self.g_tfm(x) for _ in range(self.n_global)]
                crops += [self.l_tfm(x) for _ in range(self.n_local)]
                return crops

        ds = dsets.ImageFolder(args.data, transform=MultiCropTransform(global_crop, local_crop))
        collate_fn = MultiViewCollate()

    return ds, collate_fn


# --------------------- MAE ---------------------
class MAEWrapper(nn.Module):
    """Uses timm's Masked Autoencoder ViT with a built-in loss head."""
    def __init__(self, model_name="mae_vit_base_patch16", mask_ratio=0.75, img_size=224):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            img_size=img_size,
            mask_ratio=mask_ratio
        )
        # timm MAE returns (loss, pred, mask) when given images
    def forward(self, x):
        return self.model(x)  # returns (loss, pred, mask)
    def backbone(self):
        # expose encoder for later linear probing
        return self.model.encoder


# --------------------- DINO ---------------------
def build_dino_models(model_name="vit_small_patch16_224"):
    backbone = timm.create_model(model_name, pretrained=False, num_classes=0)  # returns features
    feat_dim = backbone.num_features
    # projection heads (student and teacher) as in DINO
    proj = nn.Sequential(
        nn.Linear(feat_dim, 2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
        nn.GELU(),
        nn.Linear(2048, 256),
        nn.functional.normalize
    )
    student = nn.Sequential(backbone, nn.Flatten(1), nn.Linear(feat_dim, 2048), nn.GELU(),
                            nn.Linear(2048, 256))
    teacher_backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    teacher = nn.Sequential(teacher_backbone, nn.Flatten(1), nn.Linear(feat_dim, 2048), nn.GELU(),
                            nn.Linear(2048, 256))
    # init teacher as EMA copy
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.copy_(s.data)
        t.requires_grad_(False)
    return student, teacher, backbone


class DINOLoss(nn.Module):
    """Wrapper around lightly.models.loss.DINOLoss for simplicity."""
    def __init__(self, out_dim=256, warmup_teacher_temp=0.04, teacher_temp=0.07, warmup_epochs=10,
                 nepochs=100, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        from lightly.loss import DINOLoss as LDINOLoss
        self.loss = LDINOLoss(out_dim=out_dim,
                              warmup_teacher_temp=warmup_teacher_temp,
                              teacher_temp=teacher_temp,
                              warmup_teacher_temp_epochs=warmup_epochs,
                              nepochs=nepochs,
                              student_temp=student_temp,
                              center_momentum=center_momentum)
    def forward(self, s_outs, t_outs):
        return self.loss(s_outs, t_outs)


@torch.no_grad()
def update_ema(teacher, student, m):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data = t.data * m + s.data * (1.0 - m)


# --------------------- Utils ---------------------
def cosine_lr(optimizer, base_lr, epoch, max_epochs):
    for pg in optimizer.param_groups:
        pg['lr'] = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))


def save_backbone(state_dir, epoch, mode, backbone):
    path = Path(state_dir) / f"epoch_{epoch:03d}_{mode}_backbone.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state_dict(), path)
    return str(path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds, collate_fn = build_dataset(args)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    collate_fn=collate_fn)

    if args.mode == "mae":
        model_name = args.model or "mae_vit_base_patch16"
        mae = MAEWrapper(model_name=model_name, mask_ratio=args.mask_ratio, img_size=args.img_size).to(device)
        # timm MAE returns loss directly; use AdamW as in paper
        lr = args.lr or 1.5e-4
        opt = optim.AdamW(mae.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args.wd)

        print(f"Training MAE {model_name} for {args.epochs} epochs")
        for epoch in range(1, args.epochs + 1):
            mae.train()
            cosine_lr(opt, lr, epoch-1, args.epochs)
            epoch_loss, n = 0.0, 0
            for imgs, _ in dl:
                imgs = imgs.to(device, non_blocking=True)
                loss, _, _ = mae(imgs)  # timm MAE returns (loss, pred, mask)
                (loss / args.accum).backward()
                if (n + 1) % args.accum == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
                epoch_loss += loss.item(); n += 1
            ckpt = save_backbone(out, epoch, "mae", mae.backbone())
            print(f"[Epoch {epoch:03d}] loss={epoch_loss/n:.4f}  saved={ckpt}")

    else:  # DINO
        model_name = args.model or "vit_small_patch16_224"
        if lightly is None:
            raise RuntimeError("Install `lightly` for DINO: pip install lightly")
        student, teacher, backbone = build_dino_models(model_name)
        student, teacher = student.to(device), teacher.to(device)
        dino_loss = DINOLoss(nepochs=args.epochs).to(device)

        lr = args.lr or 5e-4
        opt = optim.AdamW(student.parameters(), lr=lr, weight_decay=args.wd)

        print(f"Training DINO {model_name} for {args.epochs} epochs")
        for epoch in range(1, args.epochs + 1):
            student.train(); teacher.eval()
            cosine_lr(opt, lr, epoch-1, args.epochs)
            loss_meter, n = 0.0, 0
            m = 0.996 - 0.3 * (1 + math.cos(math.pi * (epoch-1) / args.epochs)) / 2.0  # EMA schedule
            for views, _ in dl:
                # views: list of tensor crops [B,C,H,W] of length n_global + n_local
                views = [v.to(device, non_blocking=True) for v in views]
                # forward student/teacher on each view
                s_outs = [student(v) for v in views]
                with torch.no_grad():
                    t_outs = [teacher(v) for v in views[:2]]  # teacher sees global crops
                loss = dino_loss(s_outs, t_outs)
                (loss / args.accum).backward()
                if (n + 1) % args.accum == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
                loss_meter += loss.item(); n += 1
            update_ema(teacher, student, m)
            ckpt = save_backbone(out, epoch, "dino", backbone.to(device))
            print(f"[Epoch {epoch:03d}] loss={loss_meter/n:.4f}  saved={ckpt}")


if __name__ == "__main__":
    main()
