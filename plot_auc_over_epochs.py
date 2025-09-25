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
    Collect AUC data from all epoch folders containing metrics_both.json
    """
    epochs = []
    heart_aucs = []
    triangle_aucs = []
    marker_aucs = []
    effusion_aucs = []
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
            
            # Check if metrics_both.json exists
            try:
                metrics_file = os.path.join(folder_path, 'all_metrics.json')  
            except:
                metrics_file = os.path.join(folder_path, 'metrics_both.json')    
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                    try:
                        # Extract AUC values
                        heart_auc = data['heart']['eval_auc']
                        triangle_auc = data['triangle']['eval_auc']
                        heart_aucs.append(heart_auc)
                        triangle_aucs.append(triangle_auc)
                        marker_aucs.append(0.0)
                        effusion_aucs.append(0.0)
                    except:    
                        marker_auc = data['marker']['eval_auc']
                        effusion_auc = data['pleural_effusion']['eval_auc']
                        marker_aucs.append(marker_auc)
                        effusion_aucs.append(effusion_auc)
                        heart_aucs.append(0.0)
                        triangle_aucs.append(0.0)
                    
                    epochs.append(epoch_num)
                                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {metrics_file}: {e}")
    
    # Sort by epoch number
    sorted_data = sorted(zip(epochs, heart_aucs, triangle_aucs, marker_aucs, effusion_aucs))
    epochs, heart_aucs, triangle_aucs, marker_aucs, effusion_aucs = zip(*sorted_data) if sorted_data else ([], [], [], [], [])

    return list(epochs), list(heart_aucs), list(triangle_aucs), list(marker_aucs), list(effusion_aucs)

def plot_auc_comparison(epochs, heart_aucs, triangle_aucs, marker_aucs, effusion_aucs, save_path=None):
    """
    Create a plot comparing Heart and Triangle AUCs over epochs
    """
    plt.figure(figsize=(12, 8))
    
    # Plot both lines
    if sum(heart_aucs) > 0 and sum(triangle_aucs) > 0:
        plt.plot(epochs, heart_aucs, 'o-', label='Heart AUC', linewidth=2, markersize=6, color='red', alpha=0.8)
        plt.plot(epochs, triangle_aucs, 's-', label='Triangle AUC', linewidth=2, markersize=6, color='blue', alpha=0.8)
        plt.title('AUC Performance Comparison: Heart vs Triangle Tasks', fontsize=16, fontweight='bold')

    else:
        plt.plot(epochs, marker_aucs, 'o-', label='Laterality marker AUC', linewidth=2, markersize=6, color='red', alpha=0.8)
        plt.plot(epochs, effusion_aucs, 's-', label='Plaural Effusion AUC', linewidth=2, markersize=6, color='blue', alpha=0.8)
        plt.title('AUC Performance Comparison: Chest X-ray Tasks', fontsize=16, fontweight='bold')

    # Customize the plot
    plt.xlabel('Training Epoch', fontsize=14)
    plt.ylabel('AUC Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits to better show the data range
    if sum(heart_aucs) > 0 and sum(triangle_aucs) > 0:
        all_aucs = heart_aucs + triangle_aucs
    else: 
        all_aucs = marker_aucs + effusion_aucs
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=False, default="results_densenet", help="Path to the probe output directory")
    # Path to the probe output directory
    base_path = "/home/moll/encoder-eval/outputs/"
    base_path = base_path + ap.parse_args().path
    print(f"Using base path: {base_path}")
    print("Collecting AUC data from epoch folders...")
    epochs, heart_aucs, triangle_aucs, marker_aucs, effusion_aucs = collect_auc_data(base_path)
    
    if not epochs:
        print("No epoch data found!")
        return
    
    print(f"\nFound data for {len(epochs)} epochs: {sorted(epochs)}")
    print(f"Heart AUC range: {min(heart_aucs):.4f} - {max(heart_aucs):.4f}")
    print(f"Triangle AUC range: {min(triangle_aucs):.4f} - {max(triangle_aucs):.4f}")
    print(f"Marker AUC range: {min(marker_aucs):.4f} - {max(marker_aucs):.4f}")
    print(f"Effusion AUC range: {min(effusion_aucs):.4f} - {max(effusion_aucs):.4f}")

    # Create the plot
    save_path = "/home/moll/encoder-eval/auc_comparison_plot.png"
    plot_auc_comparison(epochs, heart_aucs, triangle_aucs, marker_aucs, effusion_aucs, save_path)

if __name__ == "__main__":
    main()
