"""
CLIP Training Modes on "silent-heart-dataset"
=============================================

Supported modes
---------------
1) standard        : Plain CLIP (in-batch negatives; duplicates treated as negatives)
2) collision_free  : ≤1 item per caption per batch (reduces false-negative collisions)
3) group_positive  : Multi-positive CLIP (all same-caption items are mutual positives)
4) clip_supcon     : Standard CLIP + image-image Supervised Contrastive auxiliary loss (positives by caption_id)
5) region_preserve : Standard CLIP with region-preserving augmentation (uses 'special_object' metadata if present)

Dataset
-------
Expects a Hugging Face dataset with columns at least: 'image', 'caption'.
If 'caption_id' exists it will be used; otherwise we derive it as sha1(caption).
If 'special_object' (JSON) + 'HR'/'LR' exist, region-preserving aug will keep that region with high probability.

Usage
-----
pip install torch torchvision transformers datasets pillow tqdm

python train_clip_modes.py \
  --mode standard \
  --dataset_id jomoll/silent-heart-dataset \
  --model_name openai/clip-vit-base-patch32 \
  --output_dir ./ckpt_standard \
  --epochs 10 --batch_size 256 --lr 5e-5

# Group-positive:
python train_clip_modes.py --mode group_positive --output_dir ./ckpt_group_pos

# Collision-free:
python train_clip_modes.py --mode collision_free --output_dir ./ckpt_collision_free

# CLIP + SupCon (image-image):
python train_clip_modes.py --mode clip_supcon --lambda_supcon 0.1 --output_dir ./ckpt_supcon

# Region-preserving augs:
python train_clip_modes.py --mode region_preserve --output_dir ./ckpt_region

"""

import os, math, json, argparse, time, random, hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, BatchSampler

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

def sha1_16(s: str) -> int:
    # Use only 8 hex chars (32 bits) to avoid overflow
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest()[:8], 16)

# ----------------------------
# Region-preserving crop
# ----------------------------

def region_preserving_random_resized_crop(img: Image.Image,
                                          target_xy: Optional[Tuple[float,float]],
                                          out_size: int,
                                          scale=(0.8, 1.0),
                                          ratio=(0.9, 1.1),
                                          attempts: int = 25) -> Image.Image:
    """
    Sample a RandomResizedCrop that *contains* target_xy (in pixel coords of current image) with high prob.
    If target_xy is None, behaves like a gentle RandomResizedCrop.
    """
    W, H = img.size
    area = W * H
    import random as _r

    for _ in range(attempts):
        target_area = area * _r.uniform(scale[0], scale[1])
        log_ratio = (_r.uniform(np.log(ratio[0]), np.log(ratio[1])))
        aspect = float(np.exp(log_ratio))
        w = int(round((target_area * aspect) ** 0.5))
        h = int(round((target_area / aspect) ** 0.5))
        if w <= W and h <= H:
            if target_xy is None:
                i = _r.randint(0, H - h)
                j = _r.randint(0, W - w)
            else:
                tx, ty = target_xy
                # Ensure target lies within crop [j, j+w) x [i, i+h)
                j_min = max(0, int(tx) - w + 1)
                j_max = min(int(tx), W - w)
                i_min = max(0, int(ty) - h + 1)
                i_max = min(int(ty), H - h)
                if j_min > j_max or i_min > i_max:
                    continue
                j = _r.randint(j_min, j_max)
                i = _r.randint(i_min, i_max)
            img_crop = img.crop((j, i, j + w, i + h))
            return img_crop.resize((out_size, out_size), resample=Image.BICUBIC)
    # Fallback: center crop then resize
    min_side = min(W, H)
    left = (W - min_side) // 2
    top = (H - min_side) // 2
    img_crop = img.crop((left, top, left + min_side, top + min_side))
    return img_crop.resize((out_size, out_size), resample=Image.BICUBIC)

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

class GentleTransform:
    """Gentle augment policy (used for region_preserve fallback): keep factor survival high without metadata."""
    def __init__(self, image_size: int, mean, std):
        import torchvision.transforms as T
        self.t = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=T.InterpolationMode.BICUBIC),
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
    caption_id: torch.Tensor     # [B]

def make_collate_fn():
    def collate(batch: List[Dict[str, Any]]) -> CLIPBatch:
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        caption_id = torch.tensor([b["caption_id"] for b in batch], dtype=torch.long)
        return CLIPBatch(pixel_values, input_ids, attention_mask, caption_id)
    return collate

# ----------------------------
# Batch sampler for collision-free
# ----------------------------

class CollisionFreeBatchSampler(BatchSampler):
    """
    Builds batches with <= 1 example per caption_id.
    """
    def __init__(self, caption_ids: List[int], batch_size: int, drop_last: bool = True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        # group indices by caption_id
        groups: Dict[int, List[int]] = {}
        for idx, cid in enumerate(caption_ids):
            groups.setdefault(int(cid), []).append(idx)
        self.groups = groups
        self.cids = list(groups.keys())

    def __iter__(self):
        # For each epoch: pick 1 random index per caption_id, then shuffle and batch
        picks = []
        for cid in self.cids:
            lst = self.groups[cid]
            picks.append(random.choice(lst))
        random.shuffle(picks)
        # yield in chunks
        batch = []
        for idx in picks:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.cids) // self.batch_size

# ----------------------------
# Dataset wrapper
# ----------------------------

def build_datasets(dataset_id: str,
                   tokenizer: CLIPTokenizerFast,
                   image_processor: CLIPImageProcessor,
                   image_size: int,
                   max_len: int,
                   mode: str,
                   num_proc: int = 1):
    ds = load_dataset(dataset_id, cache_dir="./hf_cache")

    mean = image_processor.image_mean
    std = image_processor.image_std

    train_tf = TrainTransform(image_size, mean, std)
    gentle_tf = GentleTransform(image_size, mean, std)
    eval_tf  = EvalTransform(image_size, mean, std)

    def get_caption_id(ex):
        if "caption_id" in ex and ex["caption_id"] is not None and ex["caption_id"] != "":
            try:
                if isinstance(ex["caption_id"], str) and len(ex["caption_id"]) > 8:
                    # Truncate long hex strings to avoid overflow
                    return int(ex["caption_id"][:8], 16)
                else:
                    return int(ex["caption_id"])
            except Exception:
                return sha1_16(ex["caption"])
        else:
            return sha1_16(ex["caption"])

    def parse_special_xy(ex, img: Image.Image):
        """
        Return (x,y) of special object in *current* image pixel coords if metadata allows it, else None.
        """
        try:
            if "special_object" in ex and ex["special_object"]:
                # may be JSON string in the HF dataset
                so = ex["special_object"]
                if isinstance(so, str):
                    so = json.loads(so)
                cx_hr, cy_hr = float(so["cx"]), float(so["cy"])
                HR = float(ex.get("HR", 448))
                # Image is already low-res; map HR->current
                W, H = img.size
                return (cx_hr / HR) * W, (cy_hr / HR) * H
        except Exception:
            pass
        return None

    def _prepare_train(examples):
        # Handle batched dict from HF datasets
        out = {"input_ids": [], "attention_mask": [], "pixel_values": [], "caption_id": []}
        is_batched = isinstance(examples["caption"], list)
        captions = examples["caption"] if is_batched else [examples["caption"]]
        images = examples["image"] if is_batched else [examples["image"]]
        # pass through extra metadata for region-preserving
        specials = []
        if "special_object" in examples:
            specials = examples["special_object"] if is_batched else [examples["special_object"]]
        HRs = examples.get("HR", [None]*len(captions)) if is_batched else [examples.get("HR", None)]
        # iterate
        for i, (cap, img) in enumerate(zip(captions, images)):
            tok = tokenizer(cap, padding="max_length", truncation=True, max_length=max_len, return_tensors=None)
            out["input_ids"].append(torch.tensor(tok["input_ids"], dtype=torch.long))
            out["attention_mask"].append(torch.tensor(tok["attention_mask"], dtype=torch.long))
            # region-preserving policy only in mode 'region_preserve'
            if mode == "region_preserve":
                xy = None
                # compose 'ex' view for this item if possible
                ex_single = {k: (v[i] if is_batched else v) for k, v in examples.items()}
                xy = parse_special_xy(ex_single, img)
                img_aug = region_preserving_random_resized_crop(img, xy, image_size, scale=(0.85,1.0), ratio=(0.9,1.1))
                # Ensure RGB before converting to tensor
                if img_aug.mode != "RGB":
                    img_aug = img_aug.convert("RGB")
                import torchvision.transforms as T
                norm = T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
                tens = T.ToTensor()(img_aug)
                out["pixel_values"].append(norm(tens))
            else:
                out["pixel_values"].append(train_tf(img))
            # caption_id
            ex_single = {k: (v[i] if is_batched else v) for k, v in examples.items()}
            cid = get_caption_id(ex_single)
            out["caption_id"].append(cid)
        return out

    def _prepare_eval(examples):
        out = {"input_ids": [], "attention_mask": [], "pixel_values": [], "caption_id": []}
        is_batched = isinstance(examples["caption"], list)
        captions = examples["caption"] if is_batched else [examples["caption"]]
        images = examples["image"] if is_batched else [examples["image"]]
        for i, (cap, img) in enumerate(zip(captions, images)):
            tok = tokenizer(cap, padding="max_length", truncation=True, max_length=max_len, return_tensors=None)
            out["input_ids"].append(torch.tensor(tok["input_ids"], dtype=torch.long))
            out["attention_mask"].append(torch.tensor(tok["attention_mask"], dtype=torch.long))
            out["pixel_values"].append(eval_tf(img))
            ex_single = {k: (v[i] if is_batched else v) for k, v in examples.items()}
            out["caption_id"].append(get_caption_id(ex_single))
        return out

    ds["train"] = ds["train"].with_transform(_prepare_train)
    if "val" in ds:
        ds["val"] = ds["val"].with_transform(_prepare_eval)
    if "test" in ds:
        ds["test"] = ds["test"].with_transform(_prepare_eval)

    return ds

# ----------------------------
# Losses
# ----------------------------

def clip_loss_standard(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """
    Standard symmetric cross-entropy with in-batch negatives.
    """
    device = logits_per_image.device
    bsz = logits_per_image.size(0)
    labels = torch.arange(bsz, device=device)
    loss_i = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)

def clip_loss_group_positive(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor, caption_id: torch.Tensor) -> torch.Tensor:
    """
    Multi-positive InfoNCE. Positives are all items with the same caption_id.
    Numerator averages exp(logits) over positives; denominator sums over all.
    """
    # mask of positives [B,B]
    pos = (caption_id[:, None] == caption_id[None, :])
    # avoid empty groups
    pos = pos.float()
    # image->text
    logsum_all_i = torch.logsumexp(logits_per_image, dim=1)              # [B]
    group_sizes = pos.sum(dim=1).clamp_min(1.0)                          # [B]
    # subtract log K to average the positives
    logsum_pos_i = torch.logsumexp(logits_per_image + (pos+1e-8).log(), dim=1) - group_sizes.log()
    loss_i = -(logsum_pos_i - logsum_all_i).mean()
    # text->image
    logsum_all_t = torch.logsumexp(logits_per_text, dim=1)
    logsum_pos_t = torch.logsumexp(logits_per_text + (pos.t()+1e-8).log(), dim=1) - group_sizes.log()
    loss_t = -(logsum_pos_t - logsum_all_t).mean()
    return 0.5 * (loss_i + loss_t)

def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Supervised Contrastive Loss over image projections z (L2-normalized).
    Positives are samples with the same label (here: caption_id). Excludes self-comparisons.
    Implementation follows Khosla et al. (SupCon) in log-softmax form.
    """
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / temperature            # [B,B]
    # mask self
    B = z.size(0)
    logits_mask = torch.ones((B, B), device=z.device) - torch.eye(B, device=z.device)
    # positives mask
    pos_mask = (labels[:, None] == labels[None, :]).float() * logits_mask
    # log-softmax over rows (excluding self)
    # For numerical stability, subtract max
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # [B,B]
    # compute mean log-likelihood over positives for each anchor
    pos_count = pos_mask.sum(dim=1)  # [B]
    loss = -(pos_mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
    # average over anchors that have at least one positive
    mask_valid = (pos_count > 0)
    if mask_valid.any():
        return loss[mask_valid].mean()
    else:
        return torch.tensor(0.0, device=z.device)

# ----------------------------
# Train / Eval
# ----------------------------

@dataclass
class TrainState:
    model: CLIPModel
    image_proj: Optional[nn.Module]
    optimizer: optim.Optimizer
    scheduler: Any
    scaler: torch.cuda.amp.GradScaler
    device: torch.device
    amp_dtype: torch.dtype

class ImageProjHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):  # x: [B, D]
        return self.net(x)

def evaluate(state: TrainState, loader, mode: str) -> Dict[str, float]:
    state.model.eval()
    tot_loss, n = 0.0, 0
    autocast = torch.cuda.amp.autocast if state.device.type == "cuda" else torch.cpu.amp.autocast
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            pixel_values = batch.pixel_values.to(state.device, non_blocking=True)
            input_ids = batch.input_ids.to(state.device, non_blocking=True)
            attention_mask = batch.attention_mask.to(state.device, non_blocking=True)
            caption_id = batch.caption_id.to(state.device, non_blocking=True)
            with autocast(dtype=state.amp_dtype):
                out = state.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
                if mode == "group_positive":
                    loss = clip_loss_group_positive(out.logits_per_image, out.logits_per_text, caption_id)
                else:
                    loss = clip_loss_standard(out.logits_per_image, out.logits_per_text)
                if mode == "clip_supcon" and state.image_proj is not None:
                    z = state.image_proj(out.image_embeds)
                    loss = loss + supcon_loss(z, caption_id) * 0.1  # small aux on val
            tot_loss += loss.item() * pixel_values.size(0)
            n += pixel_values.size(0)
    return {"val_loss": tot_loss / max(1, n)}

def build_loaders(ds, tokenizer, image_processor, args, device):
    collate = make_collate_fn()
    # Train loader
    if args.mode == "collision_free":
        # need caption_id array from raw dataset (pre-transform)
        raw_train = load_dataset(args.dataset_id, split="train")
        if "caption_id" in raw_train.column_names:
            caption_ids = raw_train["caption_id"]
            # convert hex strings to ints if needed
            caption_ids = [int(c, 16) if isinstance(c, str) and len(c)>8 else int(c) for c in caption_ids]
        else:
            caps = raw_train["caption"]
            caption_ids = [sha1_16(c) for c in caps]
        batch_sampler = CollisionFreeBatchSampler(caption_ids, args.batch_size, drop_last=True)
        train_loader = DataLoader(ds["train"], batch_sampler=batch_sampler,
                                  num_workers=args.num_workers, pin_memory=(device.type=="cuda"),
                                  collate_fn=collate)
    else:
        train_loader = DataLoader(
            ds["train"], batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type=="cuda"), collate_fn=collate, drop_last=True
        )
    # Val loader
    val_loader = None
    if "val" in ds:
        val_loader = DataLoader(
            ds["val"], batch_size=args.eval_batch_size or args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type=="cuda"), collate_fn=collate
        )
    return train_loader, val_loader

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_name)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_name)
    image_size = image_processor.size["shortest_edge"]

    ds = build_datasets(args.dataset_id, tokenizer, image_processor, image_size=image_size, max_len=args.max_len, mode=args.mode, num_proc=args.num_workers)
    train_loader, val_loader = build_loaders(ds, tokenizer, image_processor, args, device)

    model = CLIPModel.from_pretrained(args.model_name)
    model.to(device)

    image_proj = None
    if args.mode == "clip_supcon":
        in_dim = model.config.projection_dim  # dimension of image/text projection heads
        image_proj = ImageProjHead(in_dim, args.supcon_dim).to(device)

    # Optimizer: include proj head if present
    params = list(model.parameters()) + (list(image_proj.parameters()) if image_proj is not None else [])
    optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.weight_decay)

    # Scheduler
    steps_per_epoch = len(train_loader) // max(1, args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda"))
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    state = TrainState(model=model, image_proj=image_proj, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device, amp_dtype=amp_dtype)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        if image_proj is not None:
            image_proj.train()
        pbar = tqdm(train_loader, desc=f"Train e{epoch}", total=len(train_loader))
        running = 0.0
        for step, batch in enumerate(pbar, start=1):
            pixel_values = batch.pixel_values.to(device, non_blocking=True)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            caption_id = batch.caption_id.to(device, non_blocking=True)

            with autocast(dtype=amp_dtype):
                out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
                if args.mode == "group_positive":
                    loss = clip_loss_group_positive(out.logits_per_image, out.logits_per_text, caption_id)
                else:
                    loss = clip_loss_standard(out.logits_per_image, out.logits_per_text)

                if args.mode == "clip_supcon":
                    z = state.image_proj(out.image_embeds)
                    loss_sup = supcon_loss(z, caption_id, temperature=args.supcon_tau) * args.lambda_supcon
                    loss = loss + loss_sup

                # grad accumulation
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
                pbar.set_postfix({"loss": f"{running:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
                running = 0.0

        # Eval
        if val_loader is not None:
            metrics = evaluate(state, val_loader, args.mode)
            val_loss = metrics["val_loss"]
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f}")
            # Save best
            if val_loss < best_val:
                best_val = val_loss
                model.save_pretrained(os.path.join(args.output_dir, "best"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
                image_processor.save_pretrained(os.path.join(args.output_dir, "best"))
                if image_proj is not None:
                    torch.save(image_proj.state_dict(), os.path.join(args.output_dir, "best", "image_proj.pt"))
    # Reload best model before uploading to hub
    best_model = CLIPModel.from_pretrained(os.path.join(args.output_dir, "best"))
    best_tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.output_dir, "best"))
    try:
        best_image_processor = CLIPImageProcessor.from_pretrained(os.path.join(args.output_dir, "best"))
    except: best_image_processor = image_processor
    # save best model to huggingface hub
    best_model.push_to_hub(f"{args.output_dir}/best", organization="jomoll")
    best_tokenizer.push_to_hub(f"{args.output_dir}/best", organization="jomoll")
    if image_proj is not None:
        best_image_processor.push_to_hub(f"{args.output_dir}/best", organization="jomoll")

    print("Training done. Best val_loss:", best_val if val_loader is not None else "N/A")

    # Optional: dump val embeddings to probe later
    if args.dump_embeddings_for_val and val_loader is not None:
        dump_path = os.path.join(args.output_dir, "val_embeddings.pt")
        print("Dumping val embeddings to:", dump_path)
        model.eval()
        all_img, all_txt = [], []
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
    ap.add_argument("--mode", type=str, choices=["standard","collision_free","group_positive","clip_supcon","region_preserve"], default="standard")

    ap.add_argument("--dataset_id", type=str, default="jomoll/silent-heart-dataset")
    ap.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--output_dir", type=str, default="./outputs/ckpt_clip_modes")

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

    # SupCon options (mode=clip_supcon)
    ap.add_argument("--lambda_supcon", type=float, default=0.1, help="weight of image-image SupCon loss")
    ap.add_argument("--supcon_tau", type=float, default=0.1, help="temperature for SupCon")
    ap.add_argument("--supcon_dim", type=int, default=128, help="projection dim for SupCon head")

    ap.add_argument("--dump_embeddings_for_val", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)