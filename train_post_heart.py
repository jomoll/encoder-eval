#!/usr/bin/env python3
# posttrain_lora_heart.py
"""
LoRA post-training on the 'posttrain_heart_annotated' split.

- Uses load_model() from eval_heart_probe.py to restore your encoder exactly as evaluated.
- Applies PEFT LoRA to the vision side (Transformer q/k/v/out + mlp fc1/fc2, and visual_projection when present).
- Trains a small classifier head on heart label H (0=decoy, 1=heart).
- Saves only LoRA adapters and the head.

Example:
  python posttrain_lora_heart.py \
    --hf_name jomoll/silent-heart-datasetv2 \
    --split posttrain_heart_annotated \
    --model_path /path/to/checkpoint_dir  \
    --epochs 3 --batch_size 128 --lr 1e-4 \
    --out adapters_heart_lora
"""

import os, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPImageProcessor
from peft import LoraConfig, get_peft_model

# import your exact loader from EVAL script (source of truth for model shapes/types)
from eval_heart_probe import load_model as eval_load_model

# ---- small helpers ----

def get_image_features(model, pixel_values):
    """Works for both HF CLIPModel and your custom wrappers that implement get_image_features."""
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=pixel_values)
    # Fallback: try standard forward outputs
    out = model(pixel_values=pixel_values)
    if hasattr(out, "image_embeds"):
        return out.image_embeds
    raise AttributeError("Model does not expose image features.")

def guess_target_modules(model):
    """Dynamically determine LoRA targets based on actual model structure"""
    target_modules = []
    
    # Check if it's a custom vision encoder
    if hasattr(model, 'vision_model') and hasattr(model.vision_model, 'backbone'):
        # ResNet18, DenseNet121, VGG11 - target their backbone layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and 'backbone' in name:
                # Extract just the module name part
                parts = name.split('.')
                if len(parts) >= 2:
                    target_modules.append('.'.join(parts[-2:]))  # e.g., "layer1.0.conv1"
        # Also target the projection layer
        target_modules.extend(["visual_projection", "projection"])
        
    elif hasattr(model, 'vision_model') and hasattr(model.vision_model, 'res_blocks'):
        # SmallResNet/TinyResNet - target residual blocks and projection
        target_modules = [
            "vision_model.stem", "vision_model.downsample", 
            "vision_model.res_blocks", "vision_model.projection",
            "visual_projection"
        ]
        
    else:
        # Standard CLIP ViT
        target_modules = [
            "q_proj", "k_proj", "v_proj", "out_proj", 
            "fc1", "fc2", "visual_projection"
        ]
    
    return target_modules

class HFHeartDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, processor: CLIPImageProcessor, image_col="image", label_col="H"):
        self.ds = hf_split
        self.proc = processor
        self.image_col = image_col
        self.label_col = label_col
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex[self.image_col]
        pv = self.proc(images=img, return_tensors="pt")["pixel_values"][0]  # [3,H,W], normalized
        y = int(ex[self.label_col])
        return pv, y

# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_name", required=True)
    ap.add_argument("--split", default="posttrain_heart_annotated")
    ap.add_argument("--model_path", required=True, help="directory with your trained model (as used by eval_heart_probe.py)")
    ap.add_argument("--image_col", default="image")
    ap.add_argument("--label_col", default="H")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="adapters_heart_lora")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # load dataset split
    ds = load_dataset(args.hf_name, split=args.split)
    # use the same processor family the eval script expects
    try:
        processor = CLIPImageProcessor.from_pretrained(args.model_path)
    except Exception:
        processor = CLIPImageProcessor.from_pretrained("models/clip-vit-base-patch32")

    train_set = HFHeartDataset(ds, processor, image_col=args.image_col, label_col=args.label_col)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=not args.cpu, drop_last=True)

    # load model exactly as eval does (handles custom wrappers)
    model = eval_load_model(args.model_path, device)
    # freeze base params; LoRA will mark injected weights trainable
    for p in model.parameters():
        p.requires_grad_(False)

    # configure LoRA
    target_modules = guess_target_modules(model)
    print(f"LoRA target modules: {target_modules}")
    
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=target_modules,
        # Add these for better compatibility
        modules_to_save=["head"] if hasattr(model, 'head') else None,
    )
    
    try:
        model = get_peft_model(model, lora_cfg).to(device)
        print(f"LoRA applied successfully. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"LoRA application failed: {e}")
        print("Available modules:")
        for name, _ in model.named_modules():
            print(f"  {name}")
        raise

    model.train()

    # tiny head over image features (2-way)
    # infer emb dim from a single forward
    with torch.no_grad():
        pv0, _ = train_set[0]
        emb0 = get_image_features(model, pv0.unsqueeze(0).to(device))
        feat_dim = emb0.shape[-1]
    head = nn.Linear(feat_dim, 2).to(device)

    # only LoRA params + head train
    params = list([p for p in model.parameters() if p.requires_grad]) + list(head.parameters())
    opt = optim.AdamW(params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        tot, correct, tot_loss = 0, 0, 0.0
        for pixel_values, y in train_loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            y = torch.tensor(y, device=device, dtype=torch.long)

            feats = get_image_features(model, pixel_values)  # [B, D]
            logits = head(feats)

            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot_loss += float(loss.item()) * y.size(0)
        print(f"[{ep}/{args.epochs}] loss={tot_loss/tot:.4f} acc={correct/tot:.4f}")

    # save adapters and head
    os.makedirs(args.out, exist_ok=True)
    # PEFT will save only the adapter weights
    model.save_pretrained(args.out)
    torch.save(head.state_dict(), os.path.join(args.out, "head.pt"))
    print(f"Saved LoRA adapters to {args.out} and head to {args.out}/head.pt")

    # Save adapters and head with better compatibility
    os.makedirs(args.out, exist_ok=True)
    
    # Save LoRA adapters
    model.save_pretrained(args.out)
    
    # Save head
    torch.save(head.state_dict(), os.path.join(args.out, "head.pt"))
    
    # Save metadata for evaluation compatibility
    torch.save({
        'base_model_path': args.model_path,
        'lora_config': lora_cfg.to_dict(),
        'feature_dim': feat_dim,
        'target_modules': target_modules,
    }, os.path.join(args.out, "training_metadata.pt"))
    
    print(f"Saved LoRA adapters to {args.out}")
    print(f"To use for evaluation, modify eval_heart_probe.py to load LoRA adapters")

if __name__ == "__main__":
    main()
