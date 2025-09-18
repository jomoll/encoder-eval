import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MIMICDataset(Dataset):
    def __init__(self, df, base_path, processor, transform_labels=True):
        self.base_path = base_path
        self.processor = processor
        self.transform_labels = transform_labels
        
        # Load the IMAGE_FILENAMES file to map study_id to image paths
        print("Loading IMAGE_FILENAMES mapping...")
        self.study_to_images = {}
        image_filenames_path = os.path.join(base_path, "IMAGE_FILENAMES")
        
        with open(image_filenames_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('files/'):
                    # Extract study_id from path like: files/p18/p18873756/s50580836/image.jpg
                    parts = line.split('/')
                    if len(parts) >= 4 and parts[3].startswith('s'):
                        study_id = parts[3][1:]  # Remove 's' prefix
                        if study_id not in self.study_to_images:
                            self.study_to_images[study_id] = []
                        self.study_to_images[study_id].append(line)
        
        print(f"Loaded mappings for {len(self.study_to_images)} studies")
        
        # Filter out studies that don't have image files
        self.valid_indices = []
        self.valid_df = []
        
        print("Checking for valid studies...")
        for idx in tqdm(range(len(df))):
            row = df.iloc[idx]
            study_id = str(int(row['study_id']))
            
            # Check if we have images for this study_id
            if study_id in self.study_to_images:
                # Verify at least one image file exists
                image_exists = False
                for image_path in self.study_to_images[study_id]:
                    full_path = os.path.join(base_path, image_path)
                    if os.path.exists(full_path):
                        image_exists = True
                        break
                
                if image_exists:
                    self.valid_indices.append(idx)
                    self.valid_df.append(row)
        
        self.df = pd.DataFrame(self.valid_df).reset_index(drop=True)
        print(f"Found {len(self.df)} valid studies out of {len(df)} total")
        
        # Get label columns (exclude study_id)
        self.label_columns = [col for col in self.df.columns if col != 'study_id']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = str(int(row['study_id']))
        
        # Get the first available image for this study
        image_path = None
        for img_path in self.study_to_images[study_id]:
            full_path = os.path.join(self.base_path, img_path)
            if os.path.exists(full_path):
                image_path = full_path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"No valid image found for study {study_id}")
        
        # Load image as PIL Image 
        image = Image.open(image_path).convert('RGB')
        
        # Get labels
        labels = []
        for col in self.label_columns:
            label = row[col]
            if self.transform_labels:
                # Convert -1 (uncertain/NaN) to 0 (negative)
                label = 1 if label == 1 else 0
            labels.append(label)
        
        return {
            'image': image,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'study_id': study_id
        }

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.classifier(x))

class StrongerMultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

def custom_collate_fn(batch):
    """Custom collate function to handle PIL Images"""
    images = [item['image'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    study_ids = [item['study_id'] for item in batch]
    
    return {
        'image': images,  # Keep as list of PIL Images
        'labels': labels,
        'study_id': study_ids
    }

def load_model_and_processor(model_type, model_name, device):
    """Load model and processor based on type"""
    if model_type.lower() == 'clip':
        # CLIP models (like MedSigLIP)
        model = AutoModel.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    elif model_type.lower() == 'dino':
        # DINO models
        model = AutoModel.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def extract_embeddings_universal(model, processor, dataset, model_type, batch_size=32, device='cuda'):
    """Universal embedding extraction for different model types"""
    model.eval()
    embeddings = []
    labels = []
    study_ids = []
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {model_type.upper()} embeddings"):
            batch_images = batch['image']
            
            if model_type.lower() == 'clip':
                # CLIP processing
                inputs = processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                image_features = model.get_image_features(**inputs)
            elif model_type.lower() == 'dino':
                # DINO processing
                inputs = processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                image_features = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(image_features.cpu())
            labels.append(batch['labels'])
            study_ids.extend(batch['study_id'])
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return embeddings, labels, study_ids

def extract_embeddings_multilayer_universal(model, processor, dataset, model_type, batch_size=32, device='cuda', layers=None):
    """Universal multi-layer embedding extraction"""
    if layers is None:
        layers = list(range(-12, 0)) + [0]  # Default to 12 layers + final
    
    model.eval()
    all_embeddings = {f'layer_{layer}': [] for layer in layers}
    labels = []
    study_ids = []
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Hook function to capture intermediate outputs
    intermediate_outputs = {}
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                intermediate_outputs[layer_idx] = output[0]
            else:
                intermediate_outputs[layer_idx] = output
        return hook
    
    # Register hooks for specified layers
    hooks = []
    
    # Get transformer layers based on model type
    if model_type.lower() == 'clip':
        transformer_layers = model.vision_model.encoder.layers
    elif model_type.lower() == 'dino':
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            transformer_layers = model.encoder.layer
        elif hasattr(model, 'vit') and hasattr(model.vit.encoder, 'layer'):
            transformer_layers = model.vit.encoder.layer
        else:
            print("Warning: Could not find transformer layers in DINO model")
            transformer_layers = []
    
    for layer_idx in layers:
        if layer_idx == 0:  # Final layer
            continue
        else:
            actual_layer_idx = layer_idx % len(transformer_layers)
            if actual_layer_idx < len(transformer_layers):
                layer = transformer_layers[actual_layer_idx]
                hook = layer.register_forward_hook(hook_fn(layer_idx))
                hooks.append(hook)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting multi-layer {model_type.upper()} embeddings"):
            batch_images = batch['image']
            
            # Clear previous outputs
            intermediate_outputs.clear()
            
            if model_type.lower() == 'clip':
                # CLIP processing
                inputs = processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass to trigger hooks
                vision_outputs = model.vision_model(**inputs)
                
                # Get final layer embedding
                if 0 in layers:
                    final_embeddings = vision_outputs.pooler_output
                    final_embeddings = final_embeddings / final_embeddings.norm(dim=-1, keepdim=True)
                    all_embeddings['layer_0'].append(final_embeddings.cpu())
                    
            elif model_type.lower() == 'dino':
                # DINO processing
                inputs = processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass to trigger hooks
                outputs = model(**inputs)
                
                # Get final layer embedding
                if 0 in layers:
                    final_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                    final_embeddings = final_embeddings / final_embeddings.norm(dim=-1, keepdim=True)
                    all_embeddings['layer_0'].append(final_embeddings.cpu())
            
            # Process intermediate layer outputs
            for layer_idx in layers:
                if layer_idx != 0 and layer_idx in intermediate_outputs:
                    layer_output = intermediate_outputs[layer_idx]
                    
                    # Extract embeddings from layer output
                    if layer_output.dim() == 3:
                        layer_embeddings = layer_output[:, 0, :]  # CLS token
                    else:
                        layer_embeddings = layer_output
                    
                    # Normalize
                    layer_embeddings = layer_embeddings / layer_embeddings.norm(dim=-1, keepdim=True)
                    all_embeddings[f'layer_{layer_idx}'].append(layer_embeddings.cpu())
            
            labels.append(batch['labels'])
            study_ids.extend(batch['study_id'])
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate all embeddings
    final_embeddings = {}
    for layer_key, emb_list in all_embeddings.items():
        if emb_list:
            final_embeddings[layer_key] = torch.cat(emb_list, dim=0)
    
    labels = torch.cat(labels, dim=0)
    
    return final_embeddings, labels, study_ids

def train_stronger_classifier(embeddings, labels, num_epochs=150, lr=0.001, device='cuda'):
    """Train stronger MLP classifier"""
    num_classes = labels.shape[1]
    input_dim = embeddings.shape[1]
    
    # Use stronger architecture
    classifier = StrongerMultiLabelClassifier(input_dim, num_classes).to(device)
    
    # Class weights
    pos_weights = []
    for i in range(num_classes):
        pos_count = labels[:, i].sum().item()
        neg_count = len(labels) - pos_count
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        pos_weights.append(pos_weight)
    
    pos_weights = torch.tensor(pos_weights).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    # Better optimizer settings
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    classifier.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Wrap with sigmoid
    class SigmoidWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return torch.sigmoid(self.model(x))
    
    return SigmoidWrapper(classifier)

def train_stronger_classifier_multilayer(embeddings_dict, labels, num_epochs=150, lr=0.001, device='cuda'):
    """Train stronger MLP classifiers on multiple layer embeddings"""
    results = {}
    
    for layer_name, embeddings in embeddings_dict.items():
        print(f"\nTraining STRONGER classifier for {layer_name}...")
        print(f"Embedding shape: {embeddings.shape}")
        
        num_classes = labels.shape[1]
        input_dim = embeddings.shape[1]
        
        # Use stronger architecture
        classifier = StrongerMultiLabelClassifier(input_dim, num_classes).to(device)
        
        # Class weights
        pos_weights = []
        for i in range(num_classes):
            pos_count = labels[:, i].sum().item()
            neg_count = len(labels) - pos_count
            pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            pos_weights.append(pos_weight)
        
        pos_weights = torch.tensor(pos_weights).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        # Better optimizer settings
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        embeddings_gpu = embeddings.to(device)
        labels_gpu = labels.to(device)
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        # Train
        classifier.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = classifier(embeddings_gpu)
            loss = criterion(outputs, labels_gpu)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Wrap with sigmoid
        class SigmoidWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return torch.sigmoid(self.model(x))
        
        results[layer_name] = SigmoidWrapper(classifier)
    
    return results
def evaluate_classifier(classifier, embeddings, labels, label_names, device='cuda', debug=False):
    """Evaluate classifier performance"""
    classifier.eval()
    embeddings = embeddings.to(device)
    
    with torch.no_grad():
        outputs = classifier(embeddings)
        predictions = (outputs > 0.5).float().cpu().numpy()
        probabilities = outputs.cpu().numpy()
    
    labels_np = labels.numpy()
    
    if debug:
        print(f"Probability statistics:")
        print(f"  Min: {probabilities.min():.4f}")
        print(f"  Max: {probabilities.max():.4f}")
        print(f"  Mean: {probabilities.mean():.4f}")
        print(f"  Std: {probabilities.std():.4f}")
        print(f"Number of positive predictions (>0.5): {predictions.sum()}")
        print(f"Label distribution:")
        for i, name in enumerate(label_names):
            pos_count = labels_np[:, i].sum()
            total_count = len(labels_np[:, i])
            print(f"  {name}: {pos_count}/{total_count} ({pos_count/total_count:.3f})")
    
    # Calculate metrics for each class
    results = {}
    for i, label_name in enumerate(label_names):
        y_true = labels_np[:, i]
        y_pred = predictions[:, i]
        y_prob = probabilities[:, i]
        
        # Use different thresholds to find optimal one
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            temp_pred = (y_prob > thresh).astype(float)
            if temp_pred.sum() > 0:  # Only if we have positive predictions
                _, _, temp_f1, _ = precision_recall_fscore_support(y_true, temp_pred, average='binary', zero_division=0)
                if temp_f1 > best_f1:
                    best_f1 = temp_f1
                    best_threshold = thresh
        
        # Use best threshold for final metrics
        y_pred_best = (y_prob > best_threshold).astype(float)
        
        # Standard accuracy
        accuracy = accuracy_score(y_true, y_pred_best)
        
        # Balanced accuracy (accounts for class imbalance)
        balanced_acc = balanced_accuracy_score(y_true, y_pred_best)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_best, average='binary', zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0  # If only one class present
        
        results[label_name] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,  # New metric
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'best_threshold': best_threshold
        }
    
    return results

def evaluate_multilayer_classifiers(classifiers_dict, embeddings_dict, labels, label_names, device='cuda'):
    """Evaluate classifiers from multiple layers"""
    layer_results = {}
    
    for layer_name, classifier in classifiers_dict.items():
        print(f"\nEvaluating {layer_name}...")
        embeddings = embeddings_dict[layer_name]
        results = evaluate_classifier(classifier, embeddings, labels, label_names, device)
        layer_results[layer_name] = results
        
        # Print summary for this layer
        avg_f1 = np.mean([r['f1'] for r in results.values()])
        avg_auc = np.mean([r['auc'] for r in results.values()])
        print(f"  Average F1: {avg_f1:.3f}, Average AUC: {avg_auc:.3f}")
    
    return layer_results


def evaluate_correlation_baseline(embeddings_dict, labels, label_names):
    """Evaluate correlation-based baselines without training classifiers"""
    print("\n" + "="*100)
    print("CORRELATION-BASED BASELINE EVALUATION")
    print("="*100)
    
    correlation_results = {}
    
    for layer_name, embeddings in embeddings_dict.items():
        print(f"\nEvaluating correlations for {layer_name}...")
        
        embeddings_np = embeddings.numpy()
        labels_np = labels.numpy()
        
        layer_correlations = {}
        
        for i, label_name in enumerate(label_names):
            y_true = labels_np[:, i]
            
            # Calculate correlations between each embedding dimension and the label
            pearson_corrs = []
            spearman_corrs = []
            
            for dim in range(embeddings_np.shape[1]):
                embedding_dim = embeddings_np[:, dim]
                
                # Pearson correlation (linear relationship)
                try:
                    pearson_r, _ = pearsonr(embedding_dim, y_true)
                    pearson_corrs.append(abs(pearson_r))  # Use absolute value
                except:
                    pearson_corrs.append(0.0)
                
                # Spearman correlation (monotonic relationship)
                try:
                    spearman_r, _ = spearmanr(embedding_dim, y_true)
                    spearman_corrs.append(abs(spearman_r))  # Use absolute value
                except:
                    spearman_corrs.append(0.0)
            
            # Summary statistics for this label
            layer_correlations[label_name] = {
                'max_pearson': max(pearson_corrs),
                'mean_pearson': np.mean(pearson_corrs),
                'max_spearman': max(spearman_corrs),
                'mean_spearman': np.mean(spearman_corrs),
                'top_k_pearson': np.mean(sorted(pearson_corrs, reverse=True)[:10]),  # Top 10 dimensions
                'top_k_spearman': np.mean(sorted(spearman_corrs, reverse=True)[:10])
            }
        
        correlation_results[layer_name] = layer_correlations
        
        # Print summary for this layer
        avg_max_pearson = np.mean([corr['max_pearson'] for corr in layer_correlations.values()])
        avg_mean_pearson = np.mean([corr['mean_pearson'] for corr in layer_correlations.values()])
        avg_top_k_pearson = np.mean([corr['top_k_pearson'] for corr in layer_correlations.values()])
        
        print(f"  Avg Max Pearson: {avg_max_pearson:.4f}")
        print(f"  Avg Mean Pearson: {avg_mean_pearson:.4f}")
        print(f"  Avg Top-10 Pearson: {avg_top_k_pearson:.4f}")
    
    return correlation_results

def evaluate_simple_threshold_baseline(embeddings_dict, labels, label_names):
    """Evaluate simple threshold-based prediction using best correlated dimensions"""
    print("\n" + "="*100)
    print("SIMPLE THRESHOLD BASELINE EVALUATION")
    print("="*100)
    
    threshold_results = {}
    
    for layer_name, embeddings in embeddings_dict.items():
        print(f"\nEvaluating threshold baseline for {layer_name}...")
        
        embeddings_np = embeddings.numpy()
        labels_np = labels.numpy()
        
        layer_results = {}
        
        for i, label_name in enumerate(label_names):
            y_true = labels_np[:, i]
            
            # Find the dimension with highest correlation
            best_corr = 0
            best_dim = 0
            best_sign = 1
            
            for dim in range(embeddings_np.shape[1]):
                embedding_dim = embeddings_np[:, dim]
                
                try:
                    corr, _ = pearsonr(embedding_dim, y_true)
                    if abs(corr) > best_corr:
                        best_corr = abs(corr)
                        best_dim = dim
                        best_sign = 1 if corr > 0 else -1
                except:
                    continue
            
            if best_corr > 0:
                # Use the best dimension for prediction
                best_embedding_dim = embeddings_np[:, best_dim] * best_sign
                
                # Find optimal threshold
                thresholds = np.percentile(best_embedding_dim, [10, 20, 30, 40, 50, 60, 70, 80, 90])
                
                best_f1 = 0
                best_threshold = np.median(best_embedding_dim)
                
                for thresh in thresholds:
                    y_pred = (best_embedding_dim > thresh).astype(int)
                    if y_pred.sum() > 0:
                        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = thresh
                
                # Final prediction with best threshold
                y_pred_best = (best_embedding_dim > best_threshold).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred_best)
                balanced_acc = balanced_accuracy_score(y_true, y_pred_best)
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_best, average='binary', zero_division=0)
                
                try:
                    auc = roc_auc_score(y_true, best_embedding_dim)
                except:
                    auc = 0.0
                
                layer_results[label_name] = {
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'best_correlation': best_corr,
                    'best_dimension': best_dim,
                    'best_threshold': best_threshold
                }
            else:
                # No meaningful correlation found
                layer_results[label_name] = {
                    'accuracy': 0.0, 'balanced_accuracy': 0.5, 'precision': 0.0,
                    'recall': 0.0, 'f1': 0.0, 'auc': 0.5,
                    'best_correlation': 0.0, 'best_dimension': -1, 'best_threshold': 0.0
                }
        
        threshold_results[layer_name] = layer_results
        
        # Print summary
        avg_f1 = np.mean([r['f1'] for r in layer_results.values()])
        avg_auc = np.mean([r['auc'] for r in layer_results.values()])
        avg_corr = np.mean([r['best_correlation'] for r in layer_results.values()])
        
        print(f"  Avg F1: {avg_f1:.3f}, Avg AUC: {avg_auc:.3f}, Avg Correlation: {avg_corr:.4f}")
    
    return threshold_results

def compare_methods(trained_results, correlation_results, threshold_results, label_names):
    """Compare trained classifiers vs correlation baselines"""
    print("\n" + "="*150)
    print("METHOD COMPARISON: TRAINED CLASSIFIERS vs CORRELATION BASELINES")
    print("="*150)
    print(f"{'Layer':<12} {'Trained F1':<12} {'Threshold F1':<13} {'F1 Gain':<10} {'Trained AUC':<12} {'Threshold AUC':<13} {'AUC Gain':<10}")
    print("-"*150)
    
    # Get layer names
    layers = list(trained_results.keys())
    
    comparison_data = {}
    
    for layer_name in layers:
        if layer_name in trained_results and layer_name in threshold_results:
            # Calculate averages for trained classifiers
            trained_f1 = np.mean([r['f1'] for r in trained_results[layer_name].values()])
            trained_auc = np.mean([r['auc'] for r in trained_results[layer_name].values()])
            
            # Calculate averages for threshold baselines
            threshold_f1 = np.mean([r['f1'] for r in threshold_results[layer_name].values()])
            threshold_auc = np.mean([r['auc'] for r in threshold_results[layer_name].values()])
            
            # Calculate gains
            f1_gain = trained_f1 - threshold_f1
            auc_gain = trained_auc - threshold_auc
            
            comparison_data[layer_name] = {
                'trained_f1': trained_f1,
                'threshold_f1': threshold_f1,
                'f1_gain': f1_gain,
                'trained_auc': trained_auc,
                'threshold_auc': threshold_auc,
                'auc_gain': auc_gain
            }
            
            # Extract layer number for display
            layer_display = layer_name.replace('layer_', 'Layer ')
            if layer_display == 'Layer 0':
                layer_display = 'Final Pool'
            
            print(f"{layer_display:<12} {trained_f1:<12.3f} {threshold_f1:<13.3f} {f1_gain:<10.3f} "
                  f"{trained_auc:<12.3f} {threshold_auc:<13.3f} {auc_gain:<10.3f}")
    
    return comparison_data

def main():
    # ===== CONFIGURATION =====
    # MODEL CONFIGURATION - Change these to switch between models
    MODEL_TYPE = 'clip'  # Options: 'clip', 'dino'
    
    # Model options for each type:
    if MODEL_TYPE.lower() == 'clip':
        # CLIP/SigLIP models
        # MODEL_NAME = "google/medsiglip-448"  # Medical CLIP
        # MODEL_NAME = "openai/clip-vit-base-patch32"  # Original CLIP
        MODEL_NAME = "openai/clip-vit-large-patch14"  # Large CLIP
        NUM_LAYERS = 12  # Typical for CLIP models
    elif MODEL_TYPE.lower() == 'dino':
        # DINO models
        # MODEL_NAME = "facebook/dinov2-base"  # DINOv2
        # MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        # MODEL_NAME = "facebook/dino-vitb16"  # Original DINO
        MODEL_NAME = "facebook/dinov2-large"  # Large DINOv2
        NUM_LAYERS = 24  # Base models, use 24 for large
    
    # Data configuration
    csv_path = "../mimic-cxr-jpg/2.1.0/mimic-cxr-2.1.0-test-set-labeled.csv"
    base_path = "../mimic-cxr-jpg/2.1.0"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Model type: {MODEL_TYPE.upper()}")
    print(f"Model name: {MODEL_NAME}")
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    df = df.fillna(-1)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of studies: {len(df)}")
    
    # Load model
    print(f"Loading {MODEL_TYPE.upper()} model...")
    model, processor = load_model_and_processor(MODEL_TYPE, MODEL_NAME, device)
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Create dataset
    dataset = MIMICDataset(df, base_path, processor)
    label_names = dataset.label_columns
    
    print(f"Label columns: {label_names}")
    
    # Extract embeddings from all layers
    print(f"Extracting multi-layer {MODEL_TYPE.upper()} embeddings from {NUM_LAYERS} layers...")
    layers_to_extract = list(range(-NUM_LAYERS, 0)) + [0]  # All layers + final
    
    print(f"Extracting from layers: {layers_to_extract}")
    
    embeddings_dict, labels, study_ids = extract_embeddings_multilayer_universal(
        model, processor, dataset, MODEL_TYPE, device=device, layers=layers_to_extract
    )
    
    print(f"{MODEL_TYPE.upper()} Embedding shapes:")
    for layer_name, emb in embeddings_dict.items():
        print(f"  {layer_name}: {emb.shape}")
    
    # ===== CORRELATION BASELINES (FULL DATASET - NO TRAINING) =====
    print("\n" + "="*100)
    print(f"RUNNING CORRELATION BASELINES ON FULL DATASET ({MODEL_TYPE.upper()})")
    print("="*100)
    
    # Use FULL dataset for correlation analysis since no training is involved
    correlation_results = evaluate_correlation_baseline(embeddings_dict, labels, label_names)
    
    # ===== THRESHOLD BASELINES (FULL DATASET - NO TRAINING) =====
    threshold_results = evaluate_simple_threshold_baseline(embeddings_dict, labels, label_names)
    
    # ===== TRAIN/VAL SPLIT FOR SUPERVISED METHODS =====
    print("\n" + "="*100)
    print("PREPARING TRAIN/VAL SPLIT FOR SUPERVISED METHODS")
    print("="*100)
    
    # Train/validation split for supervised methods only
    split_embeddings = {}
    for layer_name, embeddings in embeddings_dict.items():
        train_emb, val_emb, train_labels, val_labels = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels[:, 0]
        )
        split_embeddings[layer_name] = {
            'train': train_emb,
            'val': val_emb
        }
    
    print(f"Train set size: {train_labels.shape[0]}")
    print(f"Validation set size: {val_labels.shape[0]}")
    
    # ===== TRAINED CLASSIFIERS =====
    print("\n" + "="*100)
    print(f"TRAINING SUPERVISED CLASSIFIERS ({MODEL_TYPE.upper()})")
    print("="*100)
    
    # Train classifiers for each layer
    print(f"Training STRONGER classifiers for each {MODEL_TYPE.upper()} layer...")
    train_embeddings_dict = {k: v['train'] for k, v in split_embeddings.items()}
    classifiers = train_stronger_classifier_multilayer(train_embeddings_dict, train_labels, device=device)
    
    # Evaluate all classifiers
    print(f"\nEvaluating trained {MODEL_TYPE.upper()} classifiers...")
    val_embeddings_dict = {k: v['val'] for k, v in split_embeddings.items()}
    trained_results = evaluate_multilayer_classifiers(
        classifiers, val_embeddings_dict, val_labels, label_names, device
    )
    
    # ===== RESULTS DISPLAY =====
    sorted_layers = sorted(layers_to_extract)
    
    # 1. Display Correlation Results
    print("\n" + "="*160)
    print(f"CORRELATION BASELINE RESULTS (FULL DATASET - {MODEL_TYPE.upper()})")
    print("="*160)
    print(f"{'Layer':<12} {'Max Pearson':<12} {'Mean Pearson':<13} {'Top-10 Pearson':<15} {'Max Spearman':<13} {'Mean Spearman':<14} {'Top-10 Spearman':<15}")
    print("-"*160)
    
    correlation_summary = {}
    for layer_idx in sorted_layers:
        layer_key = f'layer_{layer_idx}'
        if layer_key in correlation_results:
            results = correlation_results[layer_key]
            avg_max_pearson = np.mean([corr['max_pearson'] for corr in results.values()])
            avg_mean_pearson = np.mean([corr['mean_pearson'] for corr in results.values()])
            avg_top_k_pearson = np.mean([corr['top_k_pearson'] for corr in results.values()])
            avg_max_spearman = np.mean([corr['max_spearman'] for corr in results.values()])
            avg_mean_spearman = np.mean([corr['mean_spearman'] for corr in results.values()])
            avg_top_k_spearman = np.mean([corr['top_k_spearman'] for corr in results.values()])
            
            correlation_summary[layer_key] = {
                'max_pearson': avg_max_pearson,
                'mean_pearson': avg_mean_pearson,
                'top_k_pearson': avg_top_k_pearson,
                'max_spearman': avg_max_spearman,
                'mean_spearman': avg_mean_spearman,
                'top_k_spearman': avg_top_k_spearman
            }
            
            layer_name = f"Layer {layer_idx}" if layer_idx != 0 else "Final Pool"
            print(f"{layer_name:<12} {avg_max_pearson:<12.4f} {avg_mean_pearson:<13.4f} {avg_top_k_pearson:<15.4f} "
                  f"{avg_max_spearman:<13.4f} {avg_mean_spearman:<14.4f} {avg_top_k_spearman:<15.4f}")
    
    # 2. Display Threshold Baseline Results
    print("\n" + "="*140)
    print(f"THRESHOLD BASELINE RESULTS (FULL DATASET - {MODEL_TYPE.upper()})")
    print("="*140)
    print(f"{'Layer':<12} {'Avg F1':<8} {'Avg AUC':<8} {'Avg Precision':<12} {'Avg Recall':<10} {'Std Accuracy':<12} {'Bal Accuracy':<12}")
    print("-"*140)
    
    threshold_summary = {}
    for layer_idx in sorted_layers:
        layer_key = f'layer_{layer_idx}'
        if layer_key in threshold_results:
            results = threshold_results[layer_key]
            avg_metrics = {
                metric: np.mean([r[metric] for r in results.values()])
                for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']
            }
            threshold_summary[layer_key] = avg_metrics
            
            layer_name = f"Layer {layer_idx}" if layer_idx != 0 else "Final Pool"
            print(f"{layer_name:<12} {avg_metrics['f1']:<8.3f} {avg_metrics['auc']:<8.3f} "
                  f"{avg_metrics['precision']:<12.3f} {avg_metrics['recall']:<10.3f} "
                  f"{avg_metrics['accuracy']:<12.3f} {avg_metrics['balanced_accuracy']:<12.3f}")
    
    # 3. Display Trained Classifier Results
    print("\n" + "="*140)
    print(f"TRAINED CLASSIFIER RESULTS (VALIDATION SET - {MODEL_TYPE.upper()})")
    print("="*140)
    print(f"{'Layer':<12} {'Avg F1':<8} {'Avg AUC':<8} {'Avg Precision':<12} {'Avg Recall':<10} {'Std Accuracy':<12} {'Bal Accuracy':<12}")
    print("-"*140)
    
    trained_summary = {}
    for layer_idx in sorted_layers:
        layer_key = f'layer_{layer_idx}'
        if layer_key in trained_results:
            results = trained_results[layer_key]
            avg_metrics = {
                metric: np.mean([r[metric] for r in results.values()])
                for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']
            }
            trained_summary[layer_key] = avg_metrics
            
            layer_name = f"Layer {layer_idx}" if layer_idx != 0 else "Final Pool"
            print(f"{layer_name:<12} {avg_metrics['f1']:<8.3f} {avg_metrics['auc']:<8.3f} "
                  f"{avg_metrics['precision']:<12.3f} {avg_metrics['recall']:<10.3f} "
                  f"{avg_metrics['accuracy']:<12.3f} {avg_metrics['balanced_accuracy']:<12.3f}")
    
    # Find best performing layers for each method
    # Best correlation layer (using max Pearson)
    best_corr_layer_pearson = None
    best_corr_score_pearson = 0
    for layer_key, metrics in correlation_summary.items():
        if metrics['max_pearson'] > best_corr_score_pearson:
            best_corr_score_pearson = metrics['max_pearson']
            best_corr_layer_pearson = layer_key.replace('layer_', '')
    
    # Best correlation layer (using max Spearman)
    best_corr_layer_spearman = None
    best_corr_score_spearman = 0
    for layer_key, metrics in correlation_summary.items():
        if metrics['max_spearman'] > best_corr_score_spearman:
            best_corr_score_spearman = metrics['max_spearman']
            best_corr_layer_spearman = layer_key.replace('layer_', '')
    
    # Best threshold baseline layer
    best_threshold_layer = None
    best_threshold_f1 = 0
    for layer_key, metrics in threshold_summary.items():
        if metrics['f1'] > best_threshold_f1:
            best_threshold_f1 = metrics['f1']
            best_threshold_layer = layer_key.replace('layer_', '')
    
    # Best trained classifier layer
    best_trained_layer = None
    best_trained_f1 = 0
    for layer_key, metrics in trained_summary.items():
        if metrics['f1'] > best_trained_f1:
            best_trained_f1 = metrics['f1']
            best_trained_layer = layer_key.replace('layer_', '')
    
    print("\n" + "="*120)
    print(f"BEST PERFORMING LAYERS BY METHOD ({MODEL_TYPE.upper()}):")
    print(f"Best Pearson Correlation: Layer {best_corr_layer_pearson} (Max Pearson: {best_corr_score_pearson:.4f})")
    print(f"Best Spearman Correlation: Layer {best_corr_layer_spearman} (Max Spearman: {best_corr_score_spearman:.4f})")
    print(f"Best Threshold Baseline: Layer {best_threshold_layer} (F1: {best_threshold_f1:.3f})")
    print(f"Best Trained Classifier: Layer {best_trained_layer} (F1: {best_trained_f1:.3f})")
    print("="*120)
    
    # Save comprehensive results
    import json
    
    all_results = {
        'model_info': {
            'model_type': MODEL_TYPE.lower(),
            'model_name': MODEL_NAME,
            'num_layers': NUM_LAYERS
        },
        'correlation_baselines': {},
        'threshold_baselines': {},
        'trained_classifiers': {},
        'summary': {
            'correlation_summary': correlation_summary,
            'threshold_summary': threshold_summary,
            'trained_summary': trained_summary
        }
    }
    
    # Format correlation results for JSON
    for layer_key, results in correlation_results.items():
        layer_metrics = {}
        for label_name, metrics in results.items():
            label_metrics = {}
            for metric_name, value in metrics.items():
                label_metrics[metric_name] = float(value)
            layer_metrics[label_name] = label_metrics
        all_results['correlation_baselines'][layer_key] = layer_metrics
    
    # Format threshold results for JSON
    for layer_key, results in threshold_results.items():
        avg_metrics = {}
        for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']:
            avg_value = np.mean([r[metric] for r in results.values()])
            avg_metrics[metric] = float(avg_value)
        
        label_details = {}
        for label_name, metrics in results.items():
            label_metrics = {}
            for metric_name, value in metrics.items():
                if isinstance(value, (np.floating, np.integer)):
                    label_metrics[metric_name] = float(value)
                elif isinstance(value, (int, float)):
                    label_metrics[metric_name] = float(value)
                else:
                    label_metrics[metric_name] = value
            label_details[label_name] = label_metrics
        
        all_results['threshold_baselines'][layer_key] = {
            'averages': avg_metrics,
            'per_label': label_details
        }
    
    # Format trained results for JSON
    for layer_key, results in trained_results.items():
        avg_metrics = {}
        for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']:
            avg_value = np.mean([r[metric] for r in results.values()])
            avg_metrics[metric] = float(avg_value)
        all_results['trained_classifiers'][layer_key] = avg_metrics
    
    # Add metadata
    all_results['metadata'] = {
        'num_layers': len(layers_to_extract),
        'layers_extracted': [int(x) for x in sorted_layers],
        'num_labels': len(label_names),
        'label_names': label_names,
        'full_dataset_size': len(labels),
        'train_size': int(train_labels.shape[0]),
        'val_size': int(val_labels.shape[0]),
        'best_correlation_layer_pearson': best_corr_layer_pearson,
        'best_correlation_score_pearson': float(best_corr_score_pearson),
        'best_correlation_layer_spearman': best_corr_layer_spearman,
        'best_correlation_score_spearman': float(best_corr_score_spearman),
        'best_threshold_layer': best_threshold_layer,
        'best_threshold_f1': float(best_threshold_f1),
        'best_trained_layer': best_trained_layer,
        'best_trained_f1': float(best_trained_f1)
    }
    
    # Convert summary dictionaries to JSON-serializable format
    for summary_key in ['correlation_summary', 'threshold_summary', 'trained_summary']:
        json_summary = {}
        for layer_key, metrics in all_results['summary'][summary_key].items():
            json_metrics = {}
            for metric_name, value in metrics.items():
                json_metrics[metric_name] = float(value)
            json_summary[layer_key] = json_metrics
        all_results['summary'][summary_key] = json_summary
    
    # Save with model-specific filename
    # Clean up model name for filename
    model_name_clean = MODEL_NAME.replace("/", "_").replace("-", "_").replace(".", "_")
    filename = f'comprehensive_layer_analysis_{MODEL_TYPE.lower()}_{model_name_clean}.json'
    
    try:
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComprehensive results saved to '{filename}'")
    except Exception as e:
        print(f"Error saving JSON: {e}")
        # Save as pickle as backup
        import pickle
        pickle_filename = filename.replace('.json', '.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Results saved as pickle backup: '{pickle_filename}'")

if __name__ == "__main__":
    main()