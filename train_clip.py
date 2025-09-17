"""
Standard CLIP Training on "silent-heart-dataset"
================================================

This script trains a *standard* CLIP (in-batch negatives, no multi-positives)
on the HF dataset at: jomoll/silent-heart-dataset

Key properties
--------------
- We deliberately do **nothing** to handle duplicate captions.
  Same-caption items in a batch become negatives for each other, as in the
  original CLIP training objective.

- Minimal, dependency-light PyTorch + Transformers pipeline with AMP.
  Single-GPU ready; no accelerate/Deepspeed required (you can add them later).

- Data pipeline uses torchvision-style train/val transforms and the
  CLIP image/token processors for normalization and tokenization.

Usage
-----
pip install torch torchvision transformers datasets pillow tqdm scikit-learn

python train_clip.py \
  --dataset_id jomoll/silent-heart-dataset \
  --model_name openai/clip-vit-base-patch32 \
  --output_dir ./ckpt_standard_clip \
  --epochs 10 --batch_size 256 --lr 5e-5

For a smaller GPU, reduce --batch_size and/or use --grad_accum_steps > 1

Evaluation
---------
The script reports the symmetric CLIP loss on val. For your study you will
likely add a linear probe for heart vs decoy on frozen embeddings; you can
export embeddings with --dump_embeddings_for_val.

"""

import os, math, json, argparse, time, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image
from tqdm import tqdm

from datasets import load_dataset
from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor

# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def human_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ----------------------------
# Data transforms
# ----------------------------

class TrainTransform:
    def __init__(self, image_size: int, mean, std):
        import torchvision.transforms as T
        self.t = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    def __call__(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.t(img)

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
# Collator
# ----------------------------

@dataclass
class CLIPBatch:
    pixel_values: torch.Tensor   # [B, 3, H, W]
    input_ids: torch.Tensor      # [B, L]
    attention_mask: torch.Tensor # [B, L]

def make_collate_fn():
    def collate(batch: List[Dict[str, Any]]) -> CLIPBatch:
        # batch is a list of dicts with 'pixel_values', 'input_ids', 'attention_mask'
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        return CLIPBatch(pixel_values, input_ids, attention_mask)
    return collate

# ----------------------------
# Dataset wrapper
# ----------------------------

def build_datasets(dataset_id: str,
                   tokenizer: CLIPTokenizerFast,
                   image_processor: CLIPImageProcessor,
                   image_size: int,
                   max_len: int,
                   num_proc: int = 1):
    ds = load_dataset(dataset_id)

    mean = image_processor.image_mean
    std = image_processor.image_std

    train_tf = TrainTransform(image_size, mean, std)
    eval_tf  = EvalTransform(image_size, mean, std)

    def _prepare_train(example):
        # tokenize caption
        tok = tokenizer(example["caption"], padding="max_length", truncation=True, max_length=max_len, return_tensors=None)
        example["input_ids"] = torch.tensor(tok["input_ids"], dtype=torch.long)
        example["attention_mask"] = torch.tensor(tok["attention_mask"], dtype=torch.long)
        # image transform
        example["pixel_values"] = train_tf(example["image"])
        return example

    def _prepare_eval(example):
        tok = tokenizer(example["caption"], padding="max_length", truncation=True, max_length=max_len, return_tensors=None)
        example["input_ids"] = torch.tensor(tok["input_ids"], dtype=torch.long)
        example["attention_mask"] = torch.tensor(tok["attention_mask"], dtype=torch.long)
        example["pixel_values"] = eval_tf(example["image"])
        return example

    ds["train"] = ds["train"].with_transform(_prepare_train)
    if "val" in ds:
        ds["val"] = ds["val"].with_transform(_prepare_eval)
    if "test" in ds:
        ds["test"] = ds["test"].with_transform(_prepare_eval)

    return ds

# ----------------------------
# Loss
# ----------------------------

def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """
    Standard symmetric cross-entropy with in-batch negatives, as in CLIP.
    Assumes logits already scaled by learned temperature (model.logit_scale).
    """
    device = logits_per_image.device
    bsz = logits_per_image.size(0)
    labels = torch.arange(bsz, device=device)
    loss_i = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)

# ----------------------------
# Train / Eval
# ----------------------------

def evaluate(model, loader, device, amp_dtype=torch.float16) -> Dict[str, float]:
    model.eval()
    tot_loss, n = 0.0, 0
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            pixel_values = batch.pixel_values.to(device, non_blocking=True)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            with autocast(dtype=amp_dtype):
                out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
                loss = clip_contrastive_loss(out.logits_per_image, out.logits_per_text)
            tot_loss += loss.item() * pixel_values.size(0)
            n += pixel_values.size(0)
    return {"val_loss": tot_loss / max(1, n)}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_name)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_name)

    ds = build_datasets(args.dataset_id, tokenizer, image_processor, image_size=image_processor.size["shortest_edge"], max_len=args.max_len, num_proc=args.num_workers)

    train_loader = DataLoader(
        ds["train"], batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type=="cuda"), collate_fn=make_collate_fn(), drop_last=True
    )

    val_loader = None
    if "val" in ds:
        val_loader = DataLoader(
            ds["val"], batch_size=args.eval_batch_size or args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type=="cuda"), collate_fn=make_collate_fn()
        )

    model = CLIPModel.from_pretrained(args.model_name)
    model.to(device)

    # Optimizer
    total_steps = math.ceil(len(train_loader) * args.epochs / max(1, args.grad_accum_steps))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
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
            pixel_values = batch.pixel_values.to(device, non_blocking=True)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)

            with autocast(dtype=amp_dtype):
                out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
                loss = clip_contrastive_loss(out.logits_per_image, out.logits_per_text) / max(1, args.grad_accum_steps)

            scaler.scale(loss).backward()
            running += loss.item()

            if step % args.grad_accum_steps == 0:
                # Clip gradients if requested
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

                pbar.set_postfix({"loss": f"{running:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
                running = 0.0

        # Eval
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
            val_loss = metrics["val_loss"]
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f}")
            # Save best
            if val_loss < best_val:
                best_val = val_loss
                model.save_pretrained(os.path.join(args.output_dir, "best"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
                image_processor.save_pretrained(os.path.join(args.output_dir, "best"))
        # Save last each epoch
        model.save_pretrained(os.path.join(args.output_dir, "last"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "last"))
        image_processor.save_pretrained(os.path.join(args.output_dir, "last"))

        # save model to huggingface hub
        model.push_to_hub(f"{args.output_dir}-epoch-{epoch}", organization="jomoll")
        tokenizer.push_to_hub(f"{args.output_dir}-epoch-{epoch}", organization="jomoll")
        image_processor.push_to_hub(f"{args.output_dir}-epoch-{epoch}", organization="jomoll")
                                    
    print("Training done. Best val_loss:", best_val if val_loader is not None else "N/A")

    # Optional: dump val embeddings to probe later
    if args.dump_embeddings_for_val and val_loader is not None:
        dump_path = os.path.join(args.output_dir, "val_embeddings.pt")
        print("Dumping val embeddings to:", dump_path)
        model.eval()
        all_img, all_txt, all_ids = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Embed Val"):
                pixel_values = batch.pixel_values.to(device)
                input_ids = batch.input_ids.to(device)
                attention_mask = batch.attention_mask.to(device)
                with autocast(dtype=amp_dtype):
                    out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
                all_img.append(out.image_embeds.detach().cpu())
                all_txt.append(out.text_embeds.detach().cpu())
        all_img = torch.cat(all_img, dim=0)
        all_txt = torch.cat(all_txt, dim=0)
        torch.save({"image_embeds": all_img, "text_embeds": all_txt}, dump_path)
        print("Saved:", dump_path)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=str, default="jomoll/silent-heart-dataset")
    ap.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--output_dir", type=str, default="./ckpt_standard_clip")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
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
    ap.add_argument("--dump_embeddings_for_val", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
