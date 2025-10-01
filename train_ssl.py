#!/usr/bin/env python3
# train_ssl_hf.py
"""
MAE and DINO training on a Hugging Face dataset with columns:
  image: PIL.Image or numpy array
  captions: string (unused here)

"""

import argparse, math
import numpy as np
from pathlib import Path
from tqdm import tqdm
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
    p.add_argument("--batch_size", type=int, default=64)  # Reduced for MAE memory usage
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
        transformed = self.transform(img)
        
        # For DINO mode with MultiViewCollate, return (views, label, filename)
        # For MAE mode, return (tensor, label)
        if isinstance(transformed, list):  # DINO multi-crop
            return transformed, 0, f"sample_{i}"  # views, label, filename
        else:  # MAE single tensor
            return transformed, 0  # tensor, label


def build_mae_transform(img_size):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # stays in [0,1] as MAE expects
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # convert grayscale to RGB
    ])


class MultiCropTransform:
    def __init__(self, img_size, n_global=2, n_local=0):  # Set n_local=0
        self.n_global, self.n_local = n_global, n_local
        self.global_t = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(23, sigma=(0.1, 2.0)),
            T.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),  # ensure RGB before ToTensor
            T.ToTensor(),
            T.Normalize(**IMAGENET_NORMALIZE),
        ])

    def __call__(self, img):
        crops = [self.global_t(img) for _ in range(self.n_global)]
        # No local crops to avoid size mismatch
        return crops


# ------------------ MAE Implementation ------------------
class MAEWrapper(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", mask_ratio=0.75, img_size=224):
        super().__init__()
        # Load base ViT model as encoder
        self.encoder = timm.create_model(model_name, pretrained=False, img_size=img_size, num_classes=0)
        self.mask_ratio = mask_ratio
        self.patch_size = 16  # Assuming patch16 model
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # MAE decoder components
        embed_dim = self.encoder.num_features
        decoder_embed_dim = 512
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        # Simple decoder
        self.decoder_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=16,
                dim_feedforward=decoder_embed_dim * 4,
                batch_first=True,
                dropout=0.0,
                activation='gelu'
            ),
            num_layers=8
        )
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * 3, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize decoder pos embed
        pos_embed = self._get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 
                                                  int(self.num_patches**.5))
        # Add CLS token position (zeros) at the beginning
        pos_embed = np.concatenate([np.zeros([1, pos_embed.shape[1]]), pos_embed], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # Initialize decoder layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed

    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb

    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """Random masking following MAE"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # Get patches and apply masking
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Add cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        
        # Predictor projection
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x

    def forward_loss(self, imgs, pred, mask):
        """Compute reconstruction loss"""
        target = self.patchify(imgs)
        
        if self.encoder.norm_pix_loss if hasattr(self.encoder, 'norm_pix_loss') else False:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches
        return loss

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
        
    def backbone(self):
        return self.encoder


# ------------------ DINO ------------------
def build_dino_student_teacher(model_name="vit_small_patch16_224"):
    # backbone that returns features
    student_backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    teacher_backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    feat_dim = student_backbone.num_features
    
    # Create identical heads for both student and teacher
    student_head = nn.Sequential(
        nn.Linear(feat_dim, 2048), 
        nn.GELU(), 
        nn.Linear(2048, 256),
        nn.LayerNorm(256)
    )
    teacher_head = nn.Sequential(
        nn.Linear(feat_dim, 2048), 
        nn.GELU(), 
        nn.Linear(2048, 256),
        nn.LayerNorm(256)
    )
    
    student = nn.Sequential(student_backbone, student_head)
    teacher = nn.Sequential(teacher_backbone, teacher_head)
    
    # Initialize teacher with student weights
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.copy_(sp.data)
        tp.requires_grad = False  # Teacher parameters don't require gradients
    
    return student, teacher


@torch.no_grad()
def update_ema(teacher, student, m):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data = tp.data * m + sp.data * (1 - m)


# ------------------ utils ------------------
def cosine_lr(optimizer, base_lr, epoch, max_epochs, warmup_epochs=10):
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    
    for pg in optimizer.param_groups:
        pg['lr'] = lr


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
    print(f"Loaded dataset {args.data} split={args.split} with {len(hf_ds)} examples")
    
    if args.mode == "mae":
        transform = build_mae_transform(args.img_size)
        ds = HFViewsDataset(hf_ds, args.image_col, transform)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)

        model_name = args.model or "vit_base_patch16_224"
        model = MAEWrapper(model_name=model_name, mask_ratio=args.mask_ratio, img_size=args.img_size).to(device)
        print(f"Loaded MAE model with encoder: {model_name}")
        
        # MAE-specific optimizer settings
        lr = args.lr or 1.5e-4
        opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args.wd)
        
        print(f"Starting MAE training for {args.epochs} epochs...")
        for epoch in tqdm(range(1, args.epochs + 1), desc="Training Epochs"):
            model.train()
            cosine_lr(opt, lr, epoch-1, args.epochs)
            loss_sum, n = 0.0, 0
            
            pbar = tqdm(dl, desc=f"Epoch {epoch}")
            for imgs, _ in pbar:
                imgs = imgs.to(device, non_blocking=True)
                loss, pred, mask = model(imgs)
                    
                (loss / args.accum).backward()
                if (n + 1) % args.accum == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                
                loss_sum += loss.item()
                n += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            avg_loss = loss_sum / n
            ckpt = save_backbone(out, epoch, "mae", model.backbone())
            print(f"[{epoch:03d}] MAE loss={avg_loss:.4f} saved={ckpt}")

    else:  # DINO
        if not _has_lightly:
            raise ImportError("pip install lightly-ai")
        transform = MultiCropTransform(args.img_size)
        ds = HFViewsDataset(hf_ds, args.image_col, transform)
        collate_fn = MultiViewCollate()
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, 
                       pin_memory=True, drop_last=True, collate_fn=collate_fn)
        
        student, teacher = build_dino_student_teacher(args.model or "vit_small_patch16_224")
        student, teacher = student.to(device), teacher.to(device)
        print(f"Built DINO student and teacher with model {args.model or 'vit_small_patch16_224'}")
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr or 1e-4, weight_decay=args.wd)
        loss_fn = LightlyDINOLoss(
            output_dim=256,
            warmup_teacher_temp=0.04,
            teacher_temp=0.04,
            warmup_teacher_temp_epochs=0,
        ).to(device)

        for epoch in tqdm(range(1, args.epochs + 1), desc="Training Epochs"):
            cosine_lr(opt, args.lr or 1e-4, epoch, args.epochs)
            student.train(); teacher.eval()
            loss_sum, n = 0, 0
            for views, _, _ in tqdm(dl, desc="Training Iterations"):
                views = [v.to(device, non_blocking=True) for v in views]
                s_out = [student(v).flatten(1) for v in views]
                with torch.no_grad():
                    t_out = [teacher(v).flatten(1) for v in views]
                
                loss = loss_fn(s_out, t_out)
                (loss / args.accum).backward()
                if (n + 1) % args.accum == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
                update_ema(teacher, student, 0.996)
                loss_sum += loss.item(); n += 1
            print(f"DINO epoch {epoch:3d} loss {loss_sum/n:.4f}")
            if epoch % 10 == 0:
                save_backbone(out, epoch, "dino", student[0])


if __name__ == "__main__":
    main()
