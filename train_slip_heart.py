#!/usr/bin/env python3
"""
SLIP Training on "silent-heart-dataset"
=======================================

This implements a minimal SLIP-style trainer:
L = L_ITC (CLIP image–text) + lambda_iic * L_IIC (image–image NT-Xent)

Dataset
-------
Expects a Hugging Face dataset with at least: 'image' (PIL or array), 'caption' (str).
Optionally contains 'val' split for evaluation.

Usage
-----
pip install torch torchvision transformers datasets pillow tqdm

python train_slip.py \
  --dataset_id data/silent-heart-dataset \
  --model_name openai/clip-vit-base-patch32 \
  --output_dir ./ckpt_slip_vitb32 \
  --epochs 100 --batch_size 1024 --lambda_iic 0.5 --freeze_text
"""

import os, math, argparse, random, time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor, CLIPConfig

# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_model_info(model, name="Model"):
    total, trainable = count_parameters(model)
    print(f"{name} Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {total - trainable:,}")

# ----------------------------
# Augmentations
# ----------------------------

class TwoViewTrainTransform:
    """Two independent views for SimCLR-style image-only loss + CLIP-normalized tensor."""
    def __init__(self, image_size: int, mean, std):
        import torchvision.transforms as T
        self.normalize = T.Normalize(mean=mean, std=std)
        # Slightly stronger than eval, milder than SimCLR default to keep small objects
        base = [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
        ]
        to_tensor = [T.ToTensor(), self.normalize]
        self.t1 = T.Compose(base + to_tensor)
        self.t2 = T.Compose(base + to_tensor)

    def __call__(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.t1(img), self.t2(img)

class EvalTransform:
    def __init__(self, image_size: int, mean, std):
        import torchvision.transforms as T
        self.t = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    def __call__(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.t(img)

# ----------------------------
# Collators
# ----------------------------

@dataclass
class SLIPBatch:
    pixel_values_v1: torch.Tensor   # [B, 3, H, W]
    pixel_values_v2: torch.Tensor   # [B, 3, H, W]
    input_ids: torch.Tensor         # [B, L]
    attention_mask: torch.Tensor    # [B, L]

def make_collate_fn(train: bool = True):
    def collate(batch: List[Dict[str, Any]]) -> SLIPBatch:
        if train:
            v1 = torch.stack([b["pixel_values_v1"] for b in batch], dim=0)
            v2 = torch.stack([b["pixel_values_v2"] for b in batch], dim=0)
        else:
            # reuse v1 slot for eval single view
            v1 = torch.stack([b["pixel_values"] for b in batch], dim=0)
            v2 = v1
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        return SLIPBatch(v1, v2, input_ids, attention_mask)
    return collate

# ----------------------------
# Dataset wrappers
# ----------------------------

def build_datasets(dataset_id: str,
                   tokenizer: CLIPTokenizerFast,
                   image_processor: CLIPImageProcessor,
                   image_size: int,
                   max_len: int,
                   num_proc: int = 1):
    ds = load_dataset(dataset_id)

    mean, std = image_processor.image_mean, image_processor.image_std
    train_tf = TwoViewTrainTransform(image_size, mean, std)
    eval_tf  = EvalTransform(image_size, mean, std)

    def _prep_train(examples):
        out = {"input_ids": [], "attention_mask": [], "pixel_values_v1": [], "pixel_values_v2": []}
        imgs = examples["image"]
        caps = examples["caption"]
        for img, cap in zip(imgs, caps):
            tok = tokenizer(cap, padding="max_length", truncation=True, max_length=max_len)
            out["input_ids"].append(torch.tensor(tok["input_ids"], dtype=torch.long))
            out["attention_mask"].append(torch.tensor(tok["attention_mask"], dtype=torch.long))
            v1, v2 = train_tf(img)
            out["pixel_values_v1"].append(v1)
            out["pixel_values_v2"].append(v2)
        return out

    def _prep_eval(examples):
        out = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        imgs = examples["image"]
        caps = examples["caption"]
        for img, cap in zip(imgs, caps):
            tok = tokenizer(cap, padding="max_length", truncation=True, max_length=max_len)
            out["input_ids"].append(torch.tensor(tok["input_ids"], dtype=torch.long))
            out["attention_mask"].append(torch.tensor(tok["attention_mask"], dtype=torch.long))
            out["pixel_values"].append(eval_tf(img))
        return out

    ds["train"] = ds["train"].with_transform(_prep_train)
    if "val" in ds:
        ds["val"] = ds["val"].with_transform(_prep_eval)
    if "test" in ds:
        ds["test"] = ds["test"].with_transform(_prep_eval)
    return ds

# ----------------------------
# Losses
# ----------------------------

def clip_itc_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy (in-batch negatives)."""
    b = logits_per_image.size(0)
    device = logits_per_image.device
    labels = torch.arange(b, device=device)
    li = F.cross_entropy(logits_per_image, labels)
    lt = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (li + lt)

def nt_xent(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                 # [2B, D]
    sim = torch.matmul(z, z.t()) / temperature     # [2B, 2B]

    # numeric stability + safe masking for fp16/bf16
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    neg_inf = torch.finfo(sim.dtype).min           # ~-65504 for fp16
    sim.fill_diagonal_(neg_inf)                    # exclude self-sim

    targets = torch.arange(b, device=z.device)
    targets = torch.cat([targets + b, targets], dim=0)  # [2B]
    loss = F.cross_entropy(sim, targets)
    return loss


# ----------------------------
# Train / Eval
# ----------------------------

def evaluate(model: CLIPModel, loader, device, amp_dtype):
    model.eval()
    tot, n = 0.0, 0
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            pv = batch.pixel_values_v1.to(device, non_blocking=True)  # single view is fine
            ids = batch.input_ids.to(device, non_blocking=True)
            am  = batch.attention_mask.to(device, non_blocking=True)
            with autocast(dtype=amp_dtype):
                out = model(pixel_values=pv, input_ids=ids, attention_mask=am, return_loss=False)
                loss = clip_itc_loss(out.logits_per_image, out.logits_per_text)
            tot += loss.item() * pv.size(0)
            n += pv.size(0)
    return {"val_clip_loss": tot / max(1, n)}

def build_loaders(ds, batch_size, eval_batch_size, num_workers, device):
    train_loader = DataLoader(
        ds["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type=="cuda"),
        collate_fn=make_collate_fn(train=True), drop_last=True
    )
    val_loader = None
    if "val" in ds:
        val_loader = DataLoader(
            ds["val"], batch_size=eval_batch_size or batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type=="cuda"),
            collate_fn=make_collate_fn(train=False)
        )
    return train_loader, val_loader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_name)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_name)
    image_size = image_processor.size["shortest_edge"]

    ds = build_datasets(args.dataset_id, tokenizer, image_processor,
                        image_size=image_size, max_len=args.max_len, num_proc=args.num_workers)
    train_loader, val_loader = build_loaders(ds, args.batch_size, args.eval_batch_size, args.num_workers, device)

    cfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel(cfg)  # randomly initialized
    if args.freeze_text:
        for p in model.text_model.parameters():
            p.requires_grad_(False)
        for p in model.text_projection.parameters():
            p.requires_grad_(False)
        print("Froze text encoder.")
    print(f"Loaded CLIP: {args.model_name}")
    print_model_info(model, "CLIP (SLIP)")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader) // max(1, args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda"))
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train e{epoch}", total=len(train_loader))
        running = 0.0

        for step, batch in enumerate(pbar, start=1):
            pv1 = batch.pixel_values_v1.to(device, non_blocking=True)
            pv2 = batch.pixel_values_v2.to(device, non_blocking=True)
            ids = batch.input_ids.to(device, non_blocking=True)
            am  = batch.attention_mask.to(device, non_blocking=True)

            with autocast(dtype=amp_dtype):
                # Forward 3 branches under matched compute:
                # image(view1), image(view2), text once
                # ITC uses view1 against text (choice arbitrary but fixed)
                out_v1 = model.get_image_features(pixel_values=pv1)
                out_v2 = model.get_image_features(pixel_values=pv2)
                txt    = model.get_text_features(input_ids=ids, attention_mask=am)

                # Normalize to the CLIP projection space
                img1 = out_v1 / out_v1.norm(p=2, dim=-1, keepdim=True)
                img2 = out_v2 / out_v2.norm(p=2, dim=-1, keepdim=True)
                txtn = txt   / txt.norm(p=2, dim=-1, keepdim=True)

                # ITC logits (use CLIP's learned logit_scale)
                logit_scale = model.logit_scale.exp()
                logits_per_text  = torch.matmul(txtn, img1.t()) * logit_scale
                logits_per_image = logits_per_text.t()

                loss_itc = clip_itc_loss(logits_per_image, logits_per_text)
                loss_iic = nt_xent(img1, img2, temperature=args.iic_tau)

                loss = loss_itc + args.lambda_iic * loss_iic
                loss = loss / max(1, args.grad_accum_steps)

            scaler.scale(loss).backward()
            running += loss.item()

            if step % args.grad_accum_steps == 0:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                pbar.set_postfix({
                    "loss": f"{running:.4f}",
                    "itc":  f"{loss_itc.item():.3f}",
                    "iic":  f"{loss_iic.item():.3f}",
                    "lr":   f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                running = 0.0

        # Eval (ITC loss only, single view)
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, amp_dtype)
            val_loss = metrics["val_clip_loss"]
            print(f"[Epoch {epoch}] val_clip_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_dir = os.path.join(args.output_dir, "best")
                os.makedirs(save_dir, exist_ok=True)
                try:
                    model.save_pretrained(save_dir)
                except Exception as e:
                    print(f"save_pretrained failed: {e}. Saving state_dict instead.")
                    torch.save({"model_state_dict": model.state_dict()}, os.path.join(save_dir, "pytorch_model.bin"))
                    model.config.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                image_processor.save_pretrained(save_dir)
                print(f"Saved best model to {save_dir}")

        # periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            try:
                model.save_pretrained(save_dir)
            except Exception as e:
                print(f"save_pretrained failed: {e}. Saving state_dict instead.")
                torch.save({"model_state_dict": model.state_dict()}, os.path.join(save_dir, "pytorch_model.bin"))
                model.config.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            image_processor.save_pretrained(save_dir)
            print(f"Saved checkpoint to {save_dir}")

    print("Training done. Best val_clip_loss:", best_val if val_loader is not None else "N/A")

# ----------------------------
# Args
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=str, default="data/silent-heart-dataset")
    ap.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--output_dir", type=str, default="./outputs/ckpt_slip")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.2)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp_dtype", type=str, choices=["fp16","bf16"], default="fp16")

    # SLIP knob
    ap.add_argument("--lambda_iic", type=float, default=0.5, help="weight for image–image NT-Xent loss")
    ap.add_argument("--iic_tau", type=float, default=0.1, help="temperature for NT-Xent")

    # misc
    ap.add_argument("--freeze_text", action="store_true", help="freeze text encoder and text projection")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
