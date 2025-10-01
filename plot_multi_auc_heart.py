#!/usr/bin/env python3
import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def collect_auc_data(base_path):
    """
    Collect AUC data from all epoch folders containing metrics files
    Returns only heart AUC data
    """
    epochs = []
    heart_aucs = []
    
    # Pattern to match epoch folders
    epoch_pattern = re.compile(r'epoch_(\d+)_task-both$')
    
    # Get all directories in the base path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory and matches the epoch pattern
        if os.path.isdir(folder_path) and epoch_pattern.match(folder_name):
            # Extract epoch number
            match = epoch_pattern.match(folder_name)
            epoch_num = int(match.group(1))
            
            # Check for metrics files
            metrics_file = None
            if os.path.exists(os.path.join(folder_path, 'all_metrics.json')):
                metrics_file = os.path.join(folder_path, 'all_metrics.json')
            elif os.path.exists(os.path.join(folder_path, 'metrics_both.json')):
                metrics_file = os.path.join(folder_path, 'metrics_both.json')
            
            if metrics_file:
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract heart AUC if it exists
                    if 'heart' in data and 'eval_auc' in data['heart']:
                        heart_auc = data['heart']['eval_auc']
                        epochs.append(epoch_num)
                        heart_aucs.append(heart_auc)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {metrics_file}: {e}")
    
    # Sort by epoch number
    if epochs:
        sorted_data = sorted(zip(epochs, heart_aucs))
        epochs, heart_aucs = zip(*sorted_data)
        return list(epochs), list(heart_aucs)
    else:
        return [], []

def collect_multi_folder_data(parent_path):
    """
    Collect heart AUC data from multiple result folders
    """
    all_data = {}
    
    if not os.path.exists(parent_path):
        print(f"Parent path does not exist: {parent_path}")
        return all_data
    
    # Find all subdirectories that contain epoch folders
    for folder_name in os.listdir(parent_path):
        folder_path = os.path.join(parent_path, folder_name)
        
        if os.path.isdir(folder_path):
            # Check if this folder contains epoch subdirectories
            has_epochs = any(
                re.match(r'epoch_(\d+)_task-both$', subdir) 
                for subdir in os.listdir(folder_path) 
                if os.path.isdir(os.path.join(folder_path, subdir))
            )
            
            if has_epochs:
                print(f"Processing folder: {folder_name}")
                epochs, heart_aucs = collect_auc_data(folder_path)
                
                if epochs:
                    all_data[folder_name] = {
                        'epochs': epochs,
                        'heart_aucs': heart_aucs
                    }
                    print(f"  Found {len(epochs)} epochs with heart AUC data")
                else:
                    print(f"  No heart AUC data found")
    
    return all_data

def plot_multi_heart_auc(all_data, save_path=None):
    """
    Create a plot with heart AUC curves from multiple folders
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors for different folders
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_data)))
    
    for i, (folder_name, data) in enumerate(all_data.items()):
        epochs = data['epochs']
        heart_aucs = data['heart_aucs']
        
        plt.plot(epochs, heart_aucs, 'o-', 
                label=f'{folder_name}', 
                linewidth=2, 
                markersize=6, 
                color=colors[i], 
                alpha=0.8)
    
    # Customize the plot
    plt.title('Heart AUC Performance Comparison Across Label Mention Probabilities', fontsize=16, fontweight='bold')
    plt.xlabel('Training Epoch', fontsize=14)
    plt.ylabel('Heart AUC Score', fontsize=14)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits based on all data
    all_aucs = []
    for data in all_data.values():
        all_aucs.extend(data['heart_aucs'])
    
    if all_aucs:
        y_min = max(0.5, min(all_aucs) - 0.05)
        y_max = min(1.0, max(all_aucs) + 0.05)
        plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot heart AUC curves from multiple experiment folders")
    parser.add_argument("--path", type=str, required=False, 
                       default="/home/moll/encoder-eval/outputs/",
                       help="Path to the parent directory containing multiple result folders")
    parser.add_argument("--save-path", type=str, required=False,
                       default="/home/moll/encoder-eval/multi_heart_auc_comparison.png",
                       help="Path to save the plot")
    
    args = parser.parse_args()
    parent_path = "/home/moll/encoder-eval/outputs_new/" + args.path
    print(f"Using parent path: {parent_path}")
    print("Collecting heart AUC data from multiple folders...")
    
    all_data = collect_multi_folder_data(parent_path)
    
    if not all_data:
        print("No heart AUC data found in any folders!")
        return
    
    print(f"\nFound heart AUC data in {len(all_data)} folders:")
    for folder_name, data in all_data.items():
        epochs = data['epochs']
        heart_aucs = data['heart_aucs']
        print(f"  {folder_name}: {len(epochs)} epochs, AUC range: {min(heart_aucs):.4f} - {max(heart_aucs):.4f}")
    
    # Create the plot
    plot_multi_heart_auc(all_data, args.save_path)

if __name__ == "__main__":
    main()