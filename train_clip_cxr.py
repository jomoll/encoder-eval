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
  --mode small_resnet \
  --output_dir ./ckpt_resnet_group \
  --epochs 100 --batch_size 1024 --from_scratch --freeze_text

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, BatchSampler

from PIL import Image
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    CLIPModel, CLIPTokenizerFast, CLIPImageProcessor,
    CLIPConfig, CLIPTextConfig, CLIPVisionConfig
)
import torchvision.models as models

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

def make_collate_fn():
    def collate(batch: List[Dict[str, Any]]) -> CLIPBatch:
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
                   mode: str,
                   num_proc: int = 1):
    ds = load_dataset(dataset_id)

    mean = image_processor.image_mean
    std = image_processor.image_std

    train_tf = TrainTransform(image_size, mean, std)
    gentle_tf = GentleTransform(image_size, mean, std)
    eval_tf  = EvalTransform(image_size, mean, std)

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
        out = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        if args.findings:
            is_batched = isinstance(examples["findings_no_pleura_no_lungs_no_cardio"], list)
            captions = examples["findings_no_pleura_no_lungs_no_cardio"] if is_batched else [examples["findings_no_pleura_no_lungs_no_cardio"]]
        else:
            is_batched = isinstance(examples["impression_section"], list)
            captions = examples["impression_section"] if is_batched else [examples["impression_section"]]
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

        return out

    def _prepare_eval(examples):
        out = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        if args.findings:
            is_batched = isinstance(examples["findings_no_pleura_no_lungs_no_cardio"], list)
            captions = examples["findings_no_pleura_no_lungs_no_cardio"] if is_batched else [examples["findings_no_pleura_no_lungs_no_cardio"]]
        else:
            is_batched = isinstance(examples["impression_section"], list)
            captions = examples["impression_section"] if is_batched else [examples["impression_section"]]
        images = examples["image"] if is_batched else [examples["image"]]
        for i, (cap, img) in enumerate(zip(captions, images)):
            tok = tokenizer(cap, padding="max_length", truncation=True, max_length=max_len, return_tensors=None)
            out["input_ids"].append(torch.tensor(tok["input_ids"], dtype=torch.long))
            out["attention_mask"].append(torch.tensor(tok["attention_mask"], dtype=torch.long))
            out["pixel_values"].append(eval_tf(img))

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

class SmallResNetVision(nn.Module):
    """Small ResNet-style vision encoder for 224x224 -> vector"""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        # Assume input is 224x224, we'll downsample to 5x5 feature maps
        # then use 4 residual blocks
        
        # Initial conv: 224x224x3 -> 56x56x64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 56x56 -> 28x28
        )
        
        # Downsample to get closer to 5x5: 28x28 -> 7x7
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 7x7 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 2 residual blocks at 4x4 resolution #4
        self.res_blocks = nn.ModuleList([
            self._make_res_block(256, 256) for _ in range(4)
        ])
        
        # Global average pooling + projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(256, hidden_size)
        
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, pixel_values):
        x = self.stem(pixel_values)  # [B, 64, 28, 28]
        x = self.downsample(x)       # [B, 256, 4, 4] (close to 5x5)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = F.relu(x + residual)  # residual connection
            
        x = self.avgpool(x)          # [B, 256, 1, 1]
        x = torch.flatten(x, 1)      # [B, 256]
        x = self.projection(x)       # [B, hidden_size]
        return x

class SmallResNetCLIP(nn.Module):
    """CLIP model with small ResNet vision encoder"""
    def __init__(self, clip_model: CLIPModel, vision_hidden_size: int = 768):
        super().__init__()
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        
        # Replace vision model with small ResNet
        self.vision_model = SmallResNetVision(vision_hidden_size)
        # Keep same projection dim as original CLIP
        self.visual_projection = nn.Linear(vision_hidden_size, clip_model.config.projection_dim)
        
        self.config = clip_model.config
        
    def get_text_features(self, input_ids=None, attention_mask=None, **kwargs):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        return text_features
        
    def get_image_features(self, pixel_values=None, **kwargs):
        vision_outputs = self.vision_model(pixel_values)
        image_features = self.visual_projection(vision_outputs)
        return image_features
    
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, return_loss=True, **kwargs):
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)
        
        # Normalize features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        
        # Create output similar to CLIPModel
        from types import SimpleNamespace
        return SimpleNamespace(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_features,
            image_embeds=image_features,
        )

class TinyResNetVision(nn.Module):
    """Tiny ResNet-style vision encoder for 224x224 -> vector (much smaller than SmallResNet)"""
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        # Very lightweight architecture
        # Initial conv: 224x224x3 -> 112x112x32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 112x112 -> 56x56
        )
        
        # Aggressive downsampling: 56x56 -> 7x7
        self.downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 56x56 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 28x28 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Only 2 tiny residual blocks at 7x7 resolution
        self.res_blocks = nn.ModuleList([
            self._make_res_block(128, 128) for _ in range(2)
        ])
        
        # Global average pooling + projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(128, hidden_size)
        
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, pixel_values):
        x = self.stem(pixel_values)       # [B, 32, 56, 56]
        x = self.downsample(x)            # [B, 128, 7, 7]
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = F.relu(x + residual)      # residual connection
            
        x = self.avgpool(x)               # [B, 128, 1, 1]
        x = torch.flatten(x, 1)           # [B, 128]
        x = self.projection(x)            # [B, hidden_size]
        return x

class TinyResNetCLIP(nn.Module):
    """CLIP model with tiny ResNet vision encoder"""
    def __init__(self, clip_model: CLIPModel, vision_hidden_size: int = 256):
        super().__init__()
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        
        # Replace vision model with tiny ResNet
        self.vision_model = TinyResNetVision(vision_hidden_size)
        # Keep same projection dim as original CLIP
        self.visual_projection = nn.Linear(vision_hidden_size, clip_model.config.projection_dim)
        
        self.config = clip_model.config
        
    def get_text_features(self, input_ids=None, attention_mask=None, **kwargs):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        return text_features
        
    def get_image_features(self, pixel_values=None, **kwargs):
        vision_outputs = self.vision_model(pixel_values)
        image_features = self.visual_projection(vision_outputs)
        return image_features
    
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, return_loss=True, **kwargs):
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)
        
        # Normalize features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        
        # Create output similar to CLIPModel
        from types import SimpleNamespace
        return SimpleNamespace(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_features,
            image_embeds=image_features,
        )

class ResNet18Vision(nn.Module):
    """ResNet18 vision encoder for CLIP"""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        # Load ResNet18 architecture with random initialization (no pretrained weights)
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()  # Remove classification head
        
        # ResNet18 outputs 512 features, project to desired hidden_size
        self.projection = nn.Linear(512, hidden_size)
        
    def forward(self, pixel_values):
        # pixel_values: [B, 3, H, W]
        features = self.backbone(pixel_values)  # [B, 512]
        return self.projection(features)  # [B, hidden_size]

class DenseNet121Vision(nn.Module):
    """DenseNet121 vision encoder for CLIP"""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        # Load DenseNet121 architecture with random initialization (no pretrained weights)
        self.backbone = models.densenet121(pretrained=False)
        self.backbone.classifier = nn.Identity()  # Remove classification head
        
        # DenseNet121 outputs 1024 features, project to desired hidden_size
        self.projection = nn.Linear(1024, hidden_size)
        
    def forward(self, pixel_values):
        # pixel_values: [B, 3, H, W]
        features = self.backbone(pixel_values)  # [B, 1024]
        return self.projection(features)  # [B, hidden_size]

class VGG11Vision(nn.Module):
    """VGG11 vision encoder for CLIP"""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        # Load VGG11 architecture with random initialization (no pretrained weights)
        vgg = models.vgg11(pretrained=False)
        self.features = vgg.features  # Convolutional layers
        self.avgpool = vgg.avgpool     # Adaptive average pooling
        
        # VGG11 has a 3-layer classifier, we'll replace it with our projection
        # Original classifier input is 25088 (512 * 7 * 7)
        self.projection = nn.Linear(25088, hidden_size)
        
    def forward(self, pixel_values):
        # pixel_values: [B, 3, H, W]
        x = self.features(pixel_values)  # [B, 512, 7, 7]
        x = self.avgpool(x)              # [B, 512, 7, 7]
        x = torch.flatten(x, 1)          # [B, 25088]
        return self.projection(x)        # [B, hidden_size]

class ResNet18CLIP(nn.Module):
    """CLIP model with ResNet18 vision encoder"""
    def __init__(self, clip_model: CLIPModel, vision_hidden_size: int = 768):
        super().__init__()
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        
        # Replace vision model with ResNet18
        self.vision_model = ResNet18Vision(vision_hidden_size)
        # Keep same projection dim as original CLIP
        self.visual_projection = nn.Linear(vision_hidden_size, clip_model.config.projection_dim)
        
        self.config = clip_model.config
        
    def get_text_features(self, input_ids=None, attention_mask=None, **kwargs):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        return text_features
        
    def get_image_features(self, pixel_values=None, **kwargs):
        vision_outputs = self.vision_model(pixel_values)
        image_features = self.visual_projection(vision_outputs)
        return image_features
    
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, return_loss=True, **kwargs):
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)
        
        # Normalize features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        
        # Create output similar to CLIPModel
        from types import SimpleNamespace
        return SimpleNamespace(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_features,
            image_embeds=image_features,
        )

class DenseNet121CLIP(nn.Module):
    """CLIP model with DenseNet121 vision encoder"""
    def __init__(self, clip_model: CLIPModel, vision_hidden_size: int = 768):
        super().__init__()
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        
        # Replace vision model with DenseNet121
        self.vision_model = DenseNet121Vision(vision_hidden_size)
        # Keep same projection dim as original CLIP
        self.visual_projection = nn.Linear(vision_hidden_size, clip_model.config.projection_dim)
        
        self.config = clip_model.config
        
    def get_text_features(self, input_ids=None, attention_mask=None, **kwargs):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        return text_features
        
    def get_image_features(self, pixel_values=None, **kwargs):
        vision_outputs = self.vision_model(pixel_values)
        image_features = self.visual_projection(vision_outputs)
        return image_features
    
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, return_loss=True, **kwargs):
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)
        
        # Normalize features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        
        # Create output similar to CLIPModel
        from types import SimpleNamespace
        return SimpleNamespace(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_features,
            image_embeds=image_features,
        )

class VGG11CLIP(nn.Module):
    """CLIP model with VGG11 vision encoder"""
    def __init__(self, clip_model: CLIPModel, vision_hidden_size: int = 768):
        super().__init__()
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        
        # Replace vision model with VGG11
        self.vision_model = VGG11Vision(vision_hidden_size)
        # Keep same projection dim as original CLIP
        self.visual_projection = nn.Linear(vision_hidden_size, clip_model.config.projection_dim)
        
        self.config = clip_model.config
        
    def get_text_features(self, input_ids=None, attention_mask=None, **kwargs):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        return text_features
        
    def get_image_features(self, pixel_values=None, **kwargs):
        vision_outputs = self.vision_model(pixel_values)
        image_features = self.visual_projection(vision_outputs)
        return image_features
    
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, return_loss=True, **kwargs):
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)
        
        # Normalize features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        
        # Create output similar to CLIPModel
        from types import SimpleNamespace
        return SimpleNamespace(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_features,
            image_embeds=image_features,
        )

def evaluate(state: TrainState, loader, mode: str) -> Dict[str, float]:
    state.model.eval()
    tot_loss, n = 0.0, 0
    autocast = torch.cuda.amp.autocast if state.device.type == "cuda" else torch.cpu.amp.autocast
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            pixel_values = batch.pixel_values.to(state.device, non_blocking=True)
            input_ids = batch.input_ids.to(state.device, non_blocking=True)
            attention_mask = batch.attention_mask.to(state.device, non_blocking=True)
            with autocast(dtype=state.amp_dtype):
                out = state.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
                
                
                loss = clip_loss_standard(out.logits_per_image, out.logits_per_text)
                
            tot_loss += loss.item() * pixel_values.size(0)
            n += pixel_values.size(0)
    return {"val_loss": tot_loss / max(1, n)}

def build_loaders(ds, tokenizer, image_processor, args, device):
    collate = make_collate_fn()
    # Train loader
    
    
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

    # --- Model construction ---
    if args.mode == "small_resnet":
        base_model = CLIPModel.from_pretrained(args.model_name)
        model = SmallResNetCLIP(base_model, vision_hidden_size=512)
        print("Initialized CLIP with small ResNet vision encoder")
        print_model_info(model, "SmallResNetCLIP")
    elif args.mode == "tiny_resnet":
        base_model = CLIPModel.from_pretrained(args.model_name)
        model = TinyResNetCLIP(base_model, vision_hidden_size=256)
        print("Initialized CLIP with tiny ResNet vision encoder")
        print_model_info(model, "TinyResNetCLIP")
    elif args.mode == "resnet18":
        base_model = CLIPModel.from_pretrained(args.model_name)
        model = ResNet18CLIP(base_model, vision_hidden_size=768)
        print("Initialized CLIP with ResNet18 vision encoder")
        print_model_info(model, "ResNet18CLIP")
    elif args.mode == "densenet121":
        base_model = CLIPModel.from_pretrained(args.model_name)
        model = DenseNet121CLIP(base_model, vision_hidden_size=768)
        print("Initialized CLIP with DenseNet121 vision encoder")
        print_model_info(model, "DenseNet121CLIP")
    elif args.mode == "vgg11":
        base_model = CLIPModel.from_pretrained(args.model_name)
        model = VGG11CLIP(base_model, vision_hidden_size=768)
        print("Initialized CLIP with VGG11 vision encoder")
        print_model_info(model, "VGG11CLIP")
    elif args.from_scratch:
        # ViT-B/32-ish config, adjust if you prefer another size
        text_cfg = {
            "vocab_size": tokenizer.vocab_size,
            "max_position_embeddings": 77,
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "layer_norm_eps": 1e-5
        }
        vision_cfg = {
            "image_size": image_size,
            "patch_size": 32,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "layer_norm_eps": 1e-5
        }
        cfg = CLIPConfig(
            projection_dim=512,
            logit_scale_init_value=math.log(1/0.07),  # ≈ 2.659
            text_config=text_cfg,
            vision_config=vision_cfg
        )
        model = CLIPModel(cfg)
        print("Initialized CLIP model from scratch")
        print_model_info(model, "CLIP (from scratch)")
    else:
        model = CLIPModel.from_pretrained(args.model_name)
        print(f"Loaded pretrained CLIP model: {args.model_name}")
        print_model_info(model, "CLIP (pretrained)")

    model.to(device)

    # (optional) freeze text side
    if args.freeze_text:
        for p in model.text_model.parameters(): 
            p.requires_grad_(False)
        for p in model.text_projection.parameters(): 
            p.requires_grad_(False)
        print("Froze text encoder")
        print_model_info(model, "CLIP (after freezing text)")

    image_proj = None
    if args.mode == "clip_supcon":
        in_dim = model.config.projection_dim  # dimension of image/text projection heads
        image_proj = ImageProjHead(in_dim, args.supcon_dim).to(device)

    # Build optimizer only over trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    if args.mode == "clip_supcon" and image_proj is not None:
        params += list(image_proj.parameters())
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

    # Helper function to save model checkpoint
    def save_checkpoint(epoch, is_best=False, is_periodic=False):
        epoch=int(epoch)+1
        if is_best:
            save_dir = os.path.join(args.output_dir, "best")
        elif is_periodic:
            save_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        else:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Handle different model types
        if args.mode in ["small_resnet", "tiny_resnet", "resnet18", "densenet121", "vgg11"]:
            # Save custom model components separately
            vision_hidden_size_map = {
                "small_resnet": 512,
                "tiny_resnet": 256, 
                "resnet18": 768,
                "densenet121": 768,
                "vgg11": 768
            }
            vision_hidden_size = vision_hidden_size_map[args.mode]
            torch.save({
                'model_state_dict': model.state_dict(),
                'vision_hidden_size': vision_hidden_size,
                'mode': args.mode,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, os.path.join(save_dir, "pytorch_model.bin"))
            # Save the base components
            tokenizer.save_pretrained(save_dir)
            image_processor.save_pretrained(save_dir)
        else:
            # Standard CLIP model - save with weights_only compatibility
            try:
                model.save_pretrained(save_dir)
            except Exception as e:
                print(f"Standard save failed: {e}, trying manual save...")
                # Manual save for compatibility
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, os.path.join(save_dir, "pytorch_model.bin"))
                model.config.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            image_processor.save_pretrained(save_dir)
            
        if image_proj is not None:
            torch.save(image_proj.state_dict(), os.path.join(save_dir, "image_proj.pt"))
            
        if is_best:
            print(f"Saved best model to {save_dir}")
        elif is_periodic:
            print(f"Saved checkpoint for epoch {epoch} to {save_dir}")

    global_step = 0
    for epoch in range(args.epochs):
        if epoch == 0:
            save_checkpoint(epoch-1, is_periodic=True)
        model.train()
        if image_proj is not None:
            image_proj.train()
        pbar = tqdm(train_loader, desc=f"Train e{epoch}", total=len(train_loader))
        running = 0.0
        for step, batch in enumerate(pbar, start=1):
            pixel_values = batch.pixel_values.to(device, non_blocking=True)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)

            with autocast(dtype=amp_dtype):
                out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
                
                loss = clip_loss_standard(out.logits_per_image, out.logits_per_text)

                if args.mode == "clip_supcon":
                    z = state.image_proj(out.image_embeds)
                    loss_sup = supcon_loss(z, temperature=args.supcon_tau) * args.lambda_supcon
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
        val_loss = None
        if val_loader is not None:
            metrics = evaluate(state, val_loader, args.mode)
            val_loss = metrics["val_loss"]
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f}")
            # Save best
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(epoch, is_best=True)
        
        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch, is_periodic=True)
                    
    # Load best model for hub upload
    best_dir = os.path.join(args.output_dir, "best")
    if os.path.exists(best_dir):
        if args.mode == "small_resnet":
            # Reconstruct custom model
            base_model = CLIPModel.from_pretrained(args.model_name, weights_only=False)
            best_model = SmallResNetCLIP(base_model, vision_hidden_size=512)
            checkpoint = torch.load(os.path.join(best_dir, "pytorch_model.bin"))
            best_model.load_state_dict(checkpoint['model_state_dict'])
            
            # For hub upload, we'll save the base CLIP components and a custom config
            best_model_for_hub = base_model  # Upload the base model
            best_tokenizer = CLIPTokenizerFast.from_pretrained(best_dir)
            best_image_processor = CLIPImageProcessor.from_pretrained(best_dir)
            
            # Save custom model info in a separate file
            torch.save({
                'custom_model_type': 'small_resnet',
                'vision_hidden_size': 512,
                'small_resnet_state_dict': best_model.vision_model.state_dict(),
                'visual_projection_state_dict': best_model.visual_projection.state_dict()
            }, os.path.join(best_dir, "custom_model_info.pt"))
            
        elif args.mode == "tiny_resnet":
            # Reconstruct custom model
            base_model = CLIPModel.from_pretrained(args.model_name, weights_only=False)
            best_model = TinyResNetCLIP(base_model, vision_hidden_size=256)
            checkpoint = torch.load(os.path.join(best_dir, "pytorch_model.bin"))
            best_model.load_state_dict(checkpoint['model_state_dict'])
            
            # For hub upload, we'll save the base CLIP components and a custom config
            best_model_for_hub = base_model  # Upload the base model
            best_tokenizer = CLIPTokenizerFast.from_pretrained(best_dir)
            best_image_processor = CLIPImageProcessor.from_pretrained(best_dir)
            
            # Save custom model info in a separate file
            torch.save({
                'custom_model_type': 'tiny_resnet',
                'vision_hidden_size': 256,
                'tiny_resnet_state_dict': best_model.vision_model.state_dict(),
                'visual_projection_state_dict': best_model.visual_projection.state_dict()
            }, os.path.join(best_dir, "custom_model_info.pt"))
            
        elif args.mode == "resnet18":
            # Reconstruct custom model
            base_model = CLIPModel.from_pretrained(args.model_name, weights_only=False)
            best_model = ResNet18CLIP(base_model, vision_hidden_size=768)
            checkpoint = torch.load(os.path.join(best_dir, "pytorch_model.bin"))
            best_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Save custom model info
            torch.save({
                'custom_model_type': 'resnet18',
                'vision_hidden_size': 768,
                'resnet18_state_dict': best_model.vision_model.state_dict(),
                'visual_projection_state_dict': best_model.visual_projection.state_dict()
            }, os.path.join(best_dir, "custom_model_info.pt"))
            
            best_model_for_hub = base_model
            best_tokenizer = CLIPTokenizerFast.from_pretrained(best_dir)
            best_image_processor = CLIPImageProcessor.from_pretrained(best_dir)
            
        elif args.mode == "densenet121":
            # Reconstruct custom model
            base_model = CLIPModel.from_pretrained(args.model_name, weights_only=False)
            best_model = DenseNet121CLIP(base_model, vision_hidden_size=768)
            checkpoint = torch.load(os.path.join(best_dir, "pytorch_model.bin"))
            best_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Save custom model info
            torch.save({
                'custom_model_type': 'densenet121',
                'vision_hidden_size': 768,
                'densenet121_state_dict': best_model.vision_model.state_dict(),
                'visual_projection_state_dict': best_model.visual_projection.state_dict()
            }, os.path.join(best_dir, "custom_model_info.pt"))
            
            best_model_for_hub = base_model
            best_tokenizer = CLIPTokenizerFast.from_pretrained(best_dir)
            best_image_processor = CLIPImageProcessor.from_pretrained(best_dir)
            
        elif args.mode == "vgg11":
            # Reconstruct custom model
            base_model = CLIPModel.from_pretrained(args.model_name, weights_only=False)
            best_model = VGG11CLIP(base_model, vision_hidden_size=768)
            checkpoint = torch.load(os.path.join(best_dir, "pytorch_model.bin"))
            best_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Save custom model info
            torch.save({
                'custom_model_type': 'vgg11',
                'vision_hidden_size': 768,
                'vgg11_state_dict': best_model.vision_model.state_dict(),
                'visual_projection_state_dict': best_model.visual_projection.state_dict()
            }, os.path.join(best_dir, "custom_model_info.pt"))
            
            best_model_for_hub = base_model
            best_tokenizer = CLIPTokenizerFast.from_pretrained(best_dir)
            best_image_processor = CLIPImageProcessor.from_pretrained(best_dir)
            
        else:
            best_model_for_hub = CLIPModel.from_pretrained(best_dir)
            best_tokenizer = CLIPTokenizerFast.from_pretrained(best_dir)
            try:
                best_image_processor = CLIPImageProcessor.from_pretrained(best_dir)
            except: 
                best_image_processor = image_processor
        
        # Extract mode name from output_dir for hub naming
        output_dir_name = os.path.basename(args.output_dir.rstrip('/'))
        hub_name = f"{output_dir_name}-best"
        
        # save best model to huggingface hub
        best_model_for_hub.push_to_hub(hub_name, organization="jomoll")
        best_tokenizer.push_to_hub(hub_name, organization="jomoll")
        best_image_processor.push_to_hub(hub_name, organization="jomoll")
        
        # Upload custom model info if it exists
        if args.mode in ["small_resnet", "tiny_resnet", "resnet18", "densenet121", "vgg11"]:
            print(f"Note: Custom {args.mode} model saved locally at {best_dir}")
            print("Custom model components saved in custom_model_info.pt")

    print("Training done. Best val_loss:", best_val if val_loader is not None else "N/A")

    # Optional: dump val embeddings to probe later
    if args.dump_embeddings_for_val and val_loader is not None:
        dump_path = os.path.join(args.output_dir, "val_embeddings.pt")
        print("Dumping val embeddings to:", dump_path)
        
        # Use the actual trained model for embedding extraction
        eval_model = best_model if args.mode in ["small_resnet", "tiny_resnet"] and 'best_model' in locals() else model
        eval_model.eval()
        
        all_img, all_txt = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Embed Val"):
                pixel_values = batch.pixel_values.to(device)
                input_ids = batch.input_ids.to(device)
                attention_mask = batch.attention_mask.to(device)
                with autocast(dtype=amp_dtype):
                    out = eval_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
                all_img.append(out.image_embeds.detach().cpu())
                all_txt.append(out.text_embeds.detach().cpu())
        all_img = torch.cat(all_img, dim=0)
        all_txt = torch.cat(all_txt, dim=0)
        torch.save({"image_embeds": all_img, "text_embeds": all_txt}, dump_path)
        print("Saved:", dump_path)

def load_custom_model(model_path: str, model_name: str = "openai/clip-vit-base-patch32"):
    """Load a custom vision encoder CLIP model from saved checkpoint"""
    import os
    
    # Check if it's a custom model
    custom_info_path = os.path.join(model_path, "custom_model_info.pt")
    if os.path.exists(custom_info_path):
        # Load custom model
        custom_info = torch.load(custom_info_path, map_location='cpu')
        
        model_type = custom_info['custom_model_type']
        
        if model_type == 'small_resnet':
            # Reconstruct the model
            base_model = CLIPModel.from_pretrained(model_name)
            model = SmallResNetCLIP(base_model, vision_hidden_size=custom_info['vision_hidden_size'])
            
            # Load the custom weights
            model.vision_model.load_state_dict(custom_info['small_resnet_state_dict'])
            model.visual_projection.load_state_dict(custom_info['visual_projection_state_dict'])
            
            return model
        
        elif model_type == 'tiny_resnet':
            # Reconstruct the model
            base_model = CLIPModel.from_pretrained(model_name)
            model = TinyResNetCLIP(base_model, vision_hidden_size=custom_info['vision_hidden_size'])
            
            # Load the custom weights
            model.vision_model.load_state_dict(custom_info['tiny_resnet_state_dict'])
            model.visual_projection.load_state_dict(custom_info['visual_projection_state_dict'])
            
            return model
            
        elif model_type == 'resnet18':
            base_model = CLIPModel.from_pretrained(model_name)
            model = ResNet18CLIP(base_model, vision_hidden_size=custom_info['vision_hidden_size'])
            
            model.vision_model.load_state_dict(custom_info['resnet18_state_dict'])
            model.visual_projection.load_state_dict(custom_info['visual_projection_state_dict'])
            
            return model
            
        elif model_type == 'densenet121':
            base_model = CLIPModel.from_pretrained(model_name)
            model = DenseNet121CLIP(base_model, vision_hidden_size=custom_info['vision_hidden_size'])
            
            model.vision_model.load_state_dict(custom_info['densenet121_state_dict'])
            model.visual_projection.load_state_dict(custom_info['visual_projection_state_dict'])
            
            return model
            
        elif model_type == 'vgg11':
            base_model = CLIPModel.from_pretrained(model_name)
            model = VGG11CLIP(base_model, vision_hidden_size=custom_info['vision_hidden_size'])
            
            model.vision_model.load_state_dict(custom_info['vgg11_state_dict'])
            model.visual_projection.load_state_dict(custom_info['visual_projection_state_dict'])
            
            return model
    
    # Fall back to standard CLIP model
    return CLIPModel.from_pretrained(model_path)

def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_model_info(model, model_name="Model"):
    """Print model parameter counts"""
    total, trainable = count_parameters(model)
    print(f"{model_name} Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {total - trainable:,}")
    return total, trainable

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, 
                    choices=["standard","collision_free","group_positive","clip_supcon","region_preserve",
                            "small_resnet","tiny_resnet","resnet18","densenet121","vgg11"], 
                    default="standard")

    ap.add_argument("--dataset_id", type=str, default="jomoll/mimic-cxr-reports")
    ap.add_argument("--findings", action="store_true", help="Use 'findings' section only (if available)")
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
    
    # From scratch training
    ap.add_argument("--from_scratch", action="store_true",
                    help="Initialize CLIP with random weights instead of from_pretrained")
    ap.add_argument("--freeze_text", action="store_true",
                    help="Freeze text encoder (useful for image-only learning)")
    
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)