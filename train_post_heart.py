#!/usr/bin/env python3
# train_post_heart.py
"""
Post-training on the 'posttrain_heart_annotated' split with two modes:

1. LoRA mode: Fine-tunes vision encoder with LoRA + trains classifier head
2. CLIP mode: Continues CLIP contrastive training on extended captions

Usage:
  # LoRA fine-tuning (original approach)
  python train_post_heart.py \
    --dataset_id jomoll/silent-heart-dataset \
    --split posttrain_heart_annotated \
    --model_path /path/to/checkpoint_dir \
    --mode lora \
    --epochs 3 --batch_size 128 --lr 1e-4

  # CLIP contrastive training on extended captions
  python train_post_heart.py \
    --dataset_id jomoll/silent-heart-dataset \
    --split posttrain_heart_annotated \
    --model_path /path/to/checkpoint_dir \
    --mode clip \
    --epochs 10 --batch_size 64 --lr 5e-6
"""

import os, argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPImageProcessor, CLIPTokenizerFast
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Import model loading utilities
from eval_heart_probe import load_model as eval_load_model
from train_clip_modes import (
    clip_loss_standard, clip_loss_group_positive, 
    TrainTransform, EvalTransform, make_collate_fn,
    load_custom_model, count_parameters
)

def get_image_features(model, pixel_values):
    """Works for both HF CLIPModel and custom wrappers that implement get_image_features."""
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=pixel_values)
    # Fallback: try standard forward outputs
    out = model(pixel_values=pixel_values)
    if hasattr(out, "image_embeds"):
        return out.image_embeds
    raise AttributeError("Model does not expose image features.")

def get_text_features(model, input_ids, attention_mask):
    """Get text features from model."""
    if hasattr(model, "get_text_features"):
        return model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    # Fallback
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(out, "text_embeds"):
        return out.text_embeds
    raise AttributeError("Model does not expose text features.")

def guess_target_modules(model):
    """Dynamically determine LoRA targets based on actual model structure"""
    target_modules = set()
    
    print("Model structure for LoRA targeting:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            print(f"  {name}: {type(module).__name__}")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Target all linear layers in vision encoder
            if any(key in name.lower() for key in ['vision', 'encoder', 'transformer', 'projection']):
                # Common patterns for different architectures
                if any(pattern in name for pattern in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2', 'projection']):
                    target_modules.add(name)
                # For custom models, target linear layers in vision components
                elif 'vision' in name or 'encoder' in name:
                    target_modules.add(name)
        
        elif isinstance(module, nn.Conv2d):
            # For CNN-based vision encoders (ResNet, DenseNet, VGG)
            if 'vision' in name or 'backbone' in name:
                target_modules.add(name)
    
    target_modules = list(target_modules)
    
    # Fallback for standard CLIP
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        if hasattr(model, 'visual_projection'):
            target_modules.append("visual_projection")
    
    return target_modules

class HFHeartDataset(torch.utils.data.Dataset):
    """Dataset for LoRA mode - only needs images and heart labels"""
    def __init__(self, hf_split, processor: CLIPImageProcessor, image_col="image", label_col="H"):
        self.ds = hf_split
        self.proc = processor
        self.image_col = image_col
        self.label_col = label_col
    
    def __len__(self): 
        return len(self.ds)
    
    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex[self.image_col]
        pv = self.proc(images=img, return_tensors="pt")["pixel_values"][0]  # [3,H,W], normalized
        y = int(ex[self.label_col])
        return pv, y

class HFCLIPDataset(torch.utils.data.Dataset):
    """Dataset for CLIP mode - needs images, captions, and caption IDs"""
    def __init__(self, hf_split, image_processor: CLIPImageProcessor, tokenizer: CLIPTokenizerFast, 
                 image_size=224, max_len=77, image_col="image", caption_col="caption", augment=True):
        self.ds = hf_split
        self.image_col = image_col
        self.caption_col = caption_col
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Set up image transforms
        mean = image_processor.image_mean
        std = image_processor.image_std
        
        if augment:
            self.transform = TrainTransform(image_size, mean, std)
        else:
            self.transform = EvalTransform(image_size, mean, std)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex[self.image_col]
        caption = ex[self.caption_col]
        
        # Transform image
        pixel_values = self.transform(img)
        
        # Tokenize caption
        tok = self.tokenizer(
            caption, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors="pt"
        )
        input_ids = tok["input_ids"][0]
        attention_mask = tok["attention_mask"][0]
        
        # Generate caption_id (for group_positive loss if needed)
        import hashlib
        caption_id = int(hashlib.sha1(caption.encode('utf-8')).hexdigest()[:8], 16)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption_id": caption_id
        }

def train_lora_mode(args, device):
    """Train with LoRA fine-tuning + classifier head"""
    print("=== LoRA Fine-tuning Mode ===")
    
    # Load dataset
    ds = load_dataset(args.dataset_id, split=args.split)
    
    # Use same processor as eval script expects
    try:
        processor = CLIPImageProcessor.from_pretrained(args.model_path)
    except Exception:
        processor = CLIPImageProcessor.from_pretrained("models/clip-vit-base-patch32")

    train_set = HFHeartDataset(ds, processor, image_col=args.image_col, label_col=args.label_col)
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=(device.type=="cuda"), 
        drop_last=True
    )

    # Load model exactly as eval does
    model = eval_load_model(args.model_path, device)
    
    # Get feature dimension before applying LoRA
    with torch.no_grad():
        pv0, _ = train_set[0]
        emb0 = get_image_features(model, pv0.unsqueeze(0).to(device))
        feat_dim = emb0.shape[-1]
    
    # Freeze base params
    for p in model.parameters():
        p.requires_grad_(False)

    # Configure LoRA
    target_modules = guess_target_modules(model)
    print(f"LoRA target modules ({len(target_modules)}): {target_modules}")
    
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=None,  # Use None for custom models
        target_modules=target_modules,
    )
    
    try:
        model = get_peft_model(model, lora_cfg).to(device)
        lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA applied successfully. Trainable LoRA parameters: {lora_params:,}")
    except Exception as e:
        print(f"LoRA application failed: {e}")
        # Try fallback with minimal target modules
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and len(target_modules) < 5:
                target_modules.append(name)
        
        if target_modules:
            print(f"Trying fallback with: {target_modules}")
            lora_cfg.target_modules = target_modules
            model = get_peft_model(model, lora_cfg).to(device)
        else:
            raise

    model.train()

    # Classifier head
    head = nn.Linear(feat_dim, 2).to(device)

    # Collect trainable parameters
    lora_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    all_params = lora_params + head_params
    
    print(f"Training {len(lora_params)} LoRA params + {len(head_params)} head params = {len(all_params)} total")
    
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for ep in range(1, args.epochs + 1):
        tot, correct, tot_loss = 0, 0, 0.0
        model.train()
        
        pbar = tqdm(train_loader, desc=f"LoRA Epoch {ep}/{args.epochs}")
        for pixel_values, y in pbar:
            pixel_values = pixel_values.to(device, non_blocking=True)
            y = torch.tensor(y, device=device, dtype=torch.long)

            feats = get_image_features(model, pixel_values)  # [B, D]
            logits = head(feats)

            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tot += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot_loss += float(loss.item()) * y.size(0)
            
            pbar.set_postfix({
                "loss": f"{tot_loss/tot:.4f}", 
                "acc": f"{correct/tot:.4f}"
            })
            
        print(f"[LoRA {ep}/{args.epochs}] loss={tot_loss/tot:.4f} acc={correct/tot:.4f}")

    # Save LoRA adapters and head
    os.makedirs(args.out, exist_ok=True)
    
    try:
        model.save_pretrained(args.out)
        print(f"Saved LoRA adapters to {args.out}")
    except Exception as e:
        print(f"Failed to save LoRA adapters: {e}")
        torch.save(model.state_dict(), os.path.join(args.out, "lora_model.pt"))
    
    torch.save(head.state_dict(), os.path.join(args.out, "head.pt"))
    torch.save({
        'base_model_path': args.model_path,
        'lora_config': lora_cfg.to_dict() if hasattr(lora_cfg, 'to_dict') else vars(lora_cfg),
        'feature_dim': feat_dim,
        'target_modules': target_modules,
        'mode': 'lora'
    }, os.path.join(args.out, "training_metadata.pt"))
    
    print(f"LoRA training complete. Saved to {args.out}/")

def train_clip_mode(args, device):
    """
    CLIP contrastive training to recover lost heart representation capability.
    
    Uses extended captions that mention 'heart' to teach the vision encoder
    to extract heart-related visual features again.
    """
    print("=== CLIP Recovery Training Mode ===")
    print("Goal: Recover heart representation capability lost during pretraining")
    
    # Load dataset with heart-mentioning captions
    ds = load_dataset(args.dataset_id, split=args.split)
    
    # Load tokenizer and image processor
    try:
        tokenizer = CLIPTokenizerFast.from_pretrained(args.model_path)
        image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
    except Exception:
        tokenizer = CLIPTokenizerFast.from_pretrained("models/clip-vit-base-patch32")
        image_processor = CLIPImageProcessor.from_pretrained("models/clip-vit-base-patch32")

    # Key: Use captions that actually mention "heart"
    # This is what was missing during pretraining!
    train_set = HFCLIPDataset(
        ds, image_processor, tokenizer,
        image_size=224, max_len=77,
        image_col=args.image_col, caption_col=args.caption_col,
        augment=True
    )
    
    collate_fn = make_collate_fn()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type=="cuda"),
        collate_fn=collate_fn,
        drop_last=True
    )

    # Load model
    if os.path.exists(os.path.join(args.model_path, "custom_model_info.pt")):
        model = load_custom_model(args.model_path)
        print("Loaded custom vision encoder CLIP model")
    else:
        model = eval_load_model(args.model_path, device)
        print("Loaded standard CLIP model")
    
    model.to(device)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer - use lower LR for continued training
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    
    # Cosine scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # Training loop
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    
    for epoch in range(args.epochs):
        if epoch == 0:
            os.makedirs(f"{args.out}/epoch_{epoch}", exist_ok=True)
            checkpoint_data = {
                'model_state_dict': model.state_dict()
            }
            torch.save(checkpoint_data, os.path.join(args.out, f"epoch_{epoch}/pytorch_model.bin"))
            print(f"Saved checkpoint for epoch {epoch} to {args.out}/epoch_{epoch}/pytorch_model.bin")
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"CLIP Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            pixel_values = batch.pixel_values.to(device, non_blocking=True)
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            caption_id = batch.caption_id.to(device, non_blocking=True)

            with autocast(dtype=torch.float16):
                # Forward pass
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=False
                )
                
                # Choose loss function based on args
                if args.clip_loss_type == "group_positive":
                    loss = clip_loss_group_positive(outputs.logits_per_image, outputs.logits_per_text, caption_id)
                else:
                    loss = clip_loss_standard(outputs.logits_per_image, outputs.logits_per_text)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss/num_batches:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        print(f"[CLIP {epoch+1}/{args.epochs}] avg_loss={avg_loss:.4f}")
        # save after each epoch
        os.makedirs(f"{args.out}/epoch_{epoch+1}", exist_ok=True)

        checkpoint_data = {
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint_data, os.path.join(args.out, f"epoch_{epoch+1}/pytorch_model.bin"))
        print(f"Saved checkpoint for epoch {epoch+1} to {args.out}/epoch_{epoch+1}/pytorch_model.bin")
    # Save model
    os.makedirs(args.out, exist_ok=True )

    # Save model weights
    checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
    torch.save(checkpoint_data, os.path.join(args.out, "pytorch_model.bin"))
    print(f"Saved model weights to {args.out}/pytorch_model.bin")
    # Save tokenizer and image processor
    tokenizer.save_pretrained(args.out)
    image_processor.save_pretrained(args.out)
    
    # Save metadata
    torch.save({
        'base_model_path': args.model_path,
        'mode': 'clip',
        'clip_loss_type': args.clip_loss_type,
        'epochs': args.epochs,
        'final_lr': optimizer.param_groups[0]['lr']
    }, os.path.join(args.out, "training_metadata.pt"))
    
    # If custom model, save custom info
    custom_info_path = os.path.join(args.model_path, "custom_model_info.pt")
    if os.path.exists(custom_info_path):
        import shutil
        shutil.copy2(custom_info_path, os.path.join(args.out, "custom_model_info.pt"))
    
    print(f"CLIP training complete. Saved to {args.out}/")

def main():
    parser = argparse.ArgumentParser()
    
    # Common arguments
    parser.add_argument("--dataset_id", default="data/silent-heart-dataset", help="local path or HF hub name")
    parser.add_argument("--split", default="posttrain_heart_annotated")
    parser.add_argument("--model_path", required=True, help="directory with your trained model")
    parser.add_argument("--mode", choices=["lora", "clip"], required=True, 
                       help="lora: LoRA fine-tuning + classifier head; clip: contrastive training on extended captions")
    parser.add_argument("--image_col", default="image")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="adapters_heart_post")
    parser.add_argument("--cpu", action="store_true")
    
    # LoRA-specific arguments
    parser.add_argument("--label_col", default="H", help="Label column for LoRA mode")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # CLIP-specific arguments
    parser.add_argument("--caption_col", default="caption", help="Caption column for CLIP mode")
    parser.add_argument("--clip_loss_type", choices=["standard", "group_positive"], default="standard",
                       help="Type of CLIP loss for contrastive training")
    
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    
    # Adjust default parameters based on mode
    if args.mode == "lora":
        if args.lr == 1e-4:  # Default LR
            pass  # Keep default
        train_lora_mode(args, device)
    elif args.mode == "clip":
        if args.lr == 1e-4:  # Default LR, adjust for CLIP training
            args.lr = 5e-6  # Much lower LR for continued CLIP training
            print(f"Adjusted LR to {args.lr} for CLIP mode")
        if args.batch_size == 128:  # Default batch size
            args.batch_size = 64  # Smaller batch for CLIP
            print(f"Adjusted batch_size to {args.batch_size} for CLIP mode")
        if args.epochs == 3:  # Default epochs
            args.epochs = 10  # More epochs for CLIP
            print(f"Adjusted epochs to {args.epochs} for CLIP mode")
        train_clip_mode(args, device)

if __name__ == "__main__":
    main()
