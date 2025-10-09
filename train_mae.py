#!/usr/bin/env python3
"""
Minimal, readable Masked Autoencoder (MAE) training script in pure PyTorch.
Single file. No Lightning. No timm. Works on any image folder.

Features
- ViT-style patch embedding for encoder and a lightweight decoder
- Per-sample random masking with a fixed ratio
- Pixel reconstruction loss on normalized patches
- Cosine LR schedule with warmup
- AMP support
- Gradient accumulation
- Simple ImageFolder dataloader or Hugging Face `datasets` support (PIL in any column)

Usage
python train_mae.py \
  --data_dir /path/to/your/images \
  --epochs 100 \
  --batch_size 256 \
  --accum_steps 1 \
  --mask_ratio 0.75 \
  --img_size 224 \
  --patch_size 16 \
  --encoder_dim 768 \
  --encoder_depth 12 \
  --encoder_heads 12 \
  --decoder_dim 512 \
  --decoder_depth 8 \
  --lr 1.5e-4 \
  --weight_decay 0.05 \
  --num_workers 8 \
  --out_dir ./runs/mae_exp1

Or load directly from a Hugging Face dataset:
python train_mae.py \
  --data_dir your/dataset_or_script_name \
  --hf_split train \
  --hf_image_column Images \
  --epochs 100 --batch_size 256 --mask_ratio 0.75 --img_size 224 --patch_size 16 --out_dir ./runs/mae_hf --amp

After pretraining, you can save encoder weights and fine-tune a classifier.

This script is intentionally compact and focuses on clarity over raw speed.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils as vutils

# Optional HF datasets
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def parse_args():
    p = argparse.ArgumentParser(description="Minimal MAE Trainer")
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='./runs/mae_exp')
    p.add_argument('--hf_split', type=str, default='train', help='Split name if using HF dataset')
    p.add_argument('--hf_image_column', type=str, default='image', help='Column name containing PIL images in HF dataset')

    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--patch_size', type=int, default=16)

    p.add_argument('--encoder_dim', type=int, default=768)
    p.add_argument('--encoder_depth', type=int, default=12)
    p.add_argument('--encoder_heads', type=int, default=12)

    p.add_argument('--decoder_dim', type=int, default=512)
    p.add_argument('--decoder_depth', type=int, default=8)
    p.add_argument('--decoder_heads', type=int, default=16)

    p.add_argument('--mlp_ratio', type=float, default=4.0)
    p.add_argument('--mask_ratio', type=float, default=0.75)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--warmup_epochs', type=int, default=10)

    p.add_argument('--lr', type=float, default=1.5e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.05)

    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=1337)

    p.add_argument('--amp', action='store_true', help='Use torch.cuda.amp')
    
    # New visualization arguments
    p.add_argument('--viz_batch_size', type=int, default=8, help='Number of images to use for visualization')
    p.add_argument('--viz_mask_seed', type=int, default=42, help='Fixed seed for visualization masking (None = random)')

    return p.parse_args()


# ------------------------
# Model components
# ------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MAE(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=1,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_heads=12,
                 decoder_dim=512,
                 decoder_depth=8,
                 decoder_heads=16,
                 mlp_ratio=4.0,
                 mask_ratio=0.75):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, encoder_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_dim))
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_dim, encoder_heads, mlp_ratio) for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, decoder_heads, mlp_ratio) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.no_grad()
    def random_masking(self, x, mask_ratio):
        B, N, C = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, imgs):
        # patchify
        x = self.patch_embed(imgs)  # [B, N, C]
        B, N, C = x.shape
        # add pos and mask
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = x + self.pos_embed[:, 1:, :]
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        x_masked = torch.cat([cls_tokens, x_masked], dim=1)
        x_masked = x_masked + self.pos_embed[:, :x_masked.size(1), :]
        # encoder
        for blk in self.encoder_blocks:
            x_masked = blk(x_masked)
        x_encoded = self.encoder_norm(x_masked)
        # remove cls for decoder input
        x_encoded = x_encoded[:, 1:, :]
        # decoder: insert mask tokens
        x_dec = self.decoder_embed(x_encoded)
        mask_tokens = self.mask_token.repeat(B, N - x_encoded.size(1), 1)
        x_ = torch.cat([x_dec, mask_tokens], dim=1)  # [B, N, D]
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.size(2)))
        x_ = torch.cat([self.decoder_embed(self.cls_token.expand(B, -1, -1)), x_], dim=1)
        x_ = x_ + self.decoder_pos_embed[:, : x_.size(1), :]
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        x_ = x_[:, 1:, :]
        pred = self.decoder_pred(x_)  # [B, N, patch*patch*1]
        return pred, mask

    @torch.no_grad()
    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.img_size and H % p == 0
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    @torch.no_grad()
    def unpatchify(self, x):
        p = self.patch_size
        B, N, D = x.shape
        h = w = int(math.sqrt(N))
        x = x.reshape(B, h, w, p, p, 1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 1, h * p, w * p)
        return x


# ------------------------
# Data
# ------------------------

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_ds, image_column, transform):
        self.hf_ds = hf_ds
        self.image_column = image_column
        self.transform = transform
        # basic sanity check
        if len(hf_ds) == 0:
            raise ValueError('HF dataset split is empty')
        if image_column not in hf_ds.column_names:
            raise ValueError(f'Column {image_column} not found in HF dataset columns: {hf_ds.column_names}')

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item[self.image_column]
        # img is expected to be a PIL.Image
        img = self.transform(img)
        return img, 0  # dummy label


def build_dataloader(args):
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0),
                                    interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    if load_dataset is None:
        raise ImportError('datasets library not available. Install with: pip install datasets')
    hf_ds = load_dataset(args.data_dir, split=args.hf_split)
    ds = HFDatasetWrapper(hf_ds, args.hf_image_column, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return dl



# ------------------------
# Training utils
# ------------------------

def cosine_lr_schedule(step, total_steps, base_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_image_grid(imgs, recons, path):
    grid = torch.cat([imgs, recons], dim=0)
    grid = vutils.make_grid(grid, nrow=imgs.size(0))
    vutils.save_image(grid, path)


# ------------------------
# Main
# ------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl = build_dataloader(args)
    
    # Grab a fixed batch for visualization
    imgs_viz, _ = next(iter(dl))
    imgs_viz = imgs_viz[:args.viz_batch_size].contiguous()

    model = MAE(
        img_size=args.img_size,
        patch_size=args.patch_size,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        mlp_ratio=args.mlp_ratio,
        mask_ratio=args.mask_ratio,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    total_steps = args.epochs * len(dl)
    warmup_steps = args.warmup_epochs * len(dl)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for it, (imgs, _) in enumerate(dl):
            imgs = imgs.to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred, mask = model(imgs)
                target = model.patchify(imgs)
                loss = (F.mse_loss(pred, target, reduction='none') * mask.unsqueeze(-1)).sum() / mask.sum() / pred.size(-1)

            scaler.scale(loss / args.accum_steps).backward()

            if (it + 1) % args.accum_steps == 0:
                lr = cosine_lr_schedule(global_step, total_steps, args.lr, args.min_lr, warmup_steps)
                for pg in opt.param_groups:
                    pg['lr'] = lr
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            running += loss.item()
            if (it + 1) % 50 == 0:
                avg = running / 50
                current_lr = opt.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{args.epochs} Iter {it+1}/{len(dl)} Loss {avg:.4f} LR {current_lr:.3e}")
                running = 0.0
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            # save checkpoint and a small reconstruction grid for sanity check
            ckpt = {
                'model': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
            }
            torch.save(ckpt, out_dir / f'ckpt_{epoch:04d}.pt')
            print(f"Saved checkpoint to {out_dir / f'ckpt_{epoch:04d}.pt'}")

            # Fixed visualization batch
            model.eval()
            with torch.no_grad():
                if args.viz_mask_seed is not None:
                    torch.manual_seed(args.viz_mask_seed)
                pred, _ = model(imgs_viz.to(device, non_blocking=True))
                recons = model.unpatchify(pred).clamp(0, 1)
                save_image_grid(imgs_viz.cpu(), recons.cpu(), out_dir / f'recon_epoch_{epoch:04d}.png')

    # save encoder-only weights for fine-tuning
    torch.save(model.state_dict(), out_dir / 'mae_full.pt')
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith('decoder') and not k.startswith('mask_token') and not k.startswith('decoder_')}, out_dir / 'mae_encoder_only.pt')


if __name__ == '__main__':
    main()
