#!/usr/bin/env python3
"""
Brief MAE evaluation script - trains linear probes on frozen MAE encoder embeddings
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score

# Load the MAE model from train_mae.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_mae import MAE

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mae_checkpoint', type=str, required=True, help='Path to MAE checkpoint')
    p.add_argument('--dataset_id', type=str, default='data/silent-heart-dataset')
    p.add_argument('--split_train', type=str, default='train')
    p.add_argument('--split_eval', type=str, default='val') 
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--probe_epochs', type=int, default=100)
    p.add_argument('--probe_lr', type=float, default=1e-3)
    p.add_argument('--output_dir', type=str, default='./mae_probe_results')
    return p.parse_args()

def load_mae_model(checkpoint_path, device):
    """Load MAE model from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model args from checkpoint
    if 'args' in ckpt:
        model_args = ckpt['args']
        model = MAE(
            img_size=model_args.get('img_size', 224),
            patch_size=model_args.get('patch_size', 16),
            in_chans=1,  # Explicitly set to 1 for grayscale
            encoder_dim=model_args.get('encoder_dim', 768),
            encoder_depth=model_args.get('encoder_depth', 12),
            encoder_heads=model_args.get('encoder_heads', 12),
            decoder_dim=model_args.get('decoder_dim', 512),
            decoder_depth=model_args.get('decoder_depth', 8),
            decoder_heads=model_args.get('decoder_heads', 16),
            mlp_ratio=model_args.get('mlp_ratio', 4.0),
        )
    else:
        # Default MAE-Base config for grayscale
        model = MAE(in_chans=1)
    
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)
        
    return model

class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)
    
    def forward(self, x):
        return self.fc(x)

def extract_triangle_label(example):
    """Extract triangle label from named objects metadata"""
    try:
        # Try different possible fields for named objects
        named_objects = None
        if 'named_objects' in example:
            named_objects = example['named_objects']
        elif 'metadata' in example and example['metadata']:
            metadata = example['metadata']
            if isinstance(metadata, str):
                import json as json_lib
                metadata = json_lib.loads(metadata)
            if isinstance(metadata, dict) and 'named_objects' in metadata:
                named_objects = metadata['named_objects']
        
        if named_objects:
            if isinstance(named_objects, str):
                import json as json_lib
                named_objects = json_lib.loads(named_objects)
            
            # Check if any object has shape "triangle"
            for obj in named_objects:
                if obj.get('shape', '').lower() == 'triangle':
                    return 1
        return 0
    except:
        return 0

def extract_heart_label(example):
    """Extract heart label from H field"""
    if 'H' in example:
        return int(example['H'])
    elif 'id' in example and example['id'].endswith(('_0', '_1')):
        return int(example['id'][-1])
    else:
        return 0

class MAEDataset:
    def __init__(self, dataset_id, split, img_size=224):
        self.dataset = load_dataset(dataset_id, split=split)
        # Transform for grayscale images (single channel)
        self.transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example['image']
        
        # Convert to grayscale FIRST, before transforms
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply transforms to grayscale image
        image = self.transform(image)
        
        # Extract labels
        heart_label = extract_heart_label(example)
        triangle_label = extract_triangle_label(example)
        
        return image, heart_label, triangle_label

def get_mae_embeddings(model, dataloader, device):
    """Extract embeddings using MAE encoder"""
    model.eval()
    embeddings = []
    heart_labels = []
    triangle_labels = []
    
    with torch.no_grad():
        for batch_imgs, batch_hearts, batch_triangles in tqdm(dataloader, desc="Extracting embeddings"):
            batch_imgs = batch_imgs.to(device)
            
            # Forward through patch embedding and encoder
            x = model.patch_embed(batch_imgs)
            B, N, C = x.shape
            
            # Add position embeddings (excluding cls token position)
            x = x + model.pos_embed[:, 1:, :]
            
            # Add cls token
            cls_tokens = model.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            x = x + model.pos_embed[:, :x.size(1), :]
            
            # Forward through encoder blocks
            for blk in model.encoder_blocks:
                x = blk(x)
            x = model.encoder_norm(x)
            
            # Use cls token as representation
            cls_embeddings = x[:, 0]  # [B, encoder_dim]
            
            embeddings.append(cls_embeddings.cpu())
            heart_labels.extend(batch_hearts)
            triangle_labels.extend(batch_triangles)
    
    embeddings = torch.cat(embeddings, dim=0)
    heart_labels = torch.tensor(heart_labels)
    triangle_labels = torch.tensor(triangle_labels)
    
    return embeddings, heart_labels, triangle_labels

def train_probe(X_train, y_train, X_val, y_val, epochs=50, lr=1e-3, device='cpu'):
    """Train linear probe"""
    input_dim = X_train.shape[1]
    probe = LinearProbe(input_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    best_val_acc = 0
    for epoch in range(epochs):
        # Train
        probe.train()
        logits = probe(X_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validate
        if epoch % 10 == 0:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val)
                val_pred = val_logits.argmax(dim=1)
                val_acc = (val_pred == y_val).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                print(f"Epoch {epoch}: Train Loss {loss:.4f}, Val Acc {val_acc:.4f}")
    
    return probe

def evaluate_probe(probe, X_test, y_test, device='cpu'):
    """Evaluate probe and return metrics"""
    probe.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        logits = probe(X_test)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Probability of positive class
        preds = logits.argmax(dim=1).cpu().numpy()
        
    y_true = y_test.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(y_true, preds),
        'balanced_accuracy': balanced_accuracy_score(y_true, preds),
        'f1': f1_score(y_true, preds),
        'eval_auc': roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    return metrics

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MAE model
    print("Loading MAE model...")
    model = load_mae_model(args.mae_checkpoint, device)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MAEDataset(args.dataset_id, args.split_train, img_size=model.img_size)
    val_dataset = MAEDataset(args.dataset_id, args.split_eval, img_size=model.img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Extract embeddings
    print("Extracting training embeddings...")
    X_train, y_heart_train, y_triangle_train = get_mae_embeddings(model, train_loader, device)
    
    print("Extracting validation embeddings...")
    X_val, y_heart_val, y_triangle_val = get_mae_embeddings(model, val_loader, device)
    
    # Train probes
    print("Training heart probe...")
    heart_probe = train_probe(X_train, y_heart_train, X_val, y_heart_val, 
                             epochs=args.probe_epochs, lr=args.probe_lr, device=device)
    
    print("Training triangle probe...")
    triangle_probe = train_probe(X_train, y_triangle_train, X_val, y_triangle_val,
                                epochs=args.probe_epochs, lr=args.probe_lr, device=device)
    
    # Evaluate probes
    print("Evaluating probes...")
    heart_metrics = evaluate_probe(heart_probe, X_val, y_heart_val, device)
    triangle_metrics = evaluate_probe(triangle_probe, X_val, y_triangle_val, device)
    
    # Print results
    print("\n" + "="*50)
    print("HEART PROBE RESULTS:")
    print("="*50)
    for k, v in heart_metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\n" + "="*50)
    print("TRIANGLE PROBE RESULTS:")
    print("="*50)
    for k, v in triangle_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'heart': heart_metrics,
        'triangle': triangle_metrics,
        'dataset_info': {
            'n_train': len(train_dataset),
            'n_val': len(val_dataset)
        }
    }
    
    with open(os.path.join(args.output_dir, 'metrics_both.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/metrics_both.json")

if __name__ == '__main__':
    main()