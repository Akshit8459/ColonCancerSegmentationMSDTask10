#!/usr/bin/env python3
"""
plotting.py
=============================================================================
Automatic Plotting Engine for MSD Task10 Colon Benchmarking Suite.
Generates publication-quality plots:
- Stage 1 Screening: Validation Dice vs Wall-Clock Time & Validation Dice vs Epochs.
- Stage 2 Confirmation: 5-Fold Training Loss & Validation Dice Curves.
- Final Evaluation: OOF Dice Comparison Bar Charts & Box Plots.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_stage1_curves(stage1_dir, output_dir):
    """
    Plots Stage 1 Screening curves: Dice vs. Wall-Clock Time and Dice vs. Epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for arch_dir in sorted(os.listdir(stage1_dir)):
        arch_path = os.path.join(stage1_dir, arch_dir)
        summary_path = os.path.join(arch_path, 'fold_summary.json')
        if os.path.exists(summary_path):
            data = json.load(open(summary_path))
            history = data.get("history", [])
            if not history:
                continue
                
            epochs = [h["epoch"] for h in history]
            times = [h["elapsed_seconds"] / 60.0 for h in history] # Minutes
            dices = [h["val_dice"] for h in history]
            
            ax1.plot(times, dices, marker='o', linewidth=2, label=arch_dir)
            ax2.plot(epochs, dices, marker='s', linewidth=2, label=arch_dir)

    ax1.set_xlabel("Elapsed Wall-Clock Time (Minutes)", fontsize=12)
    ax1.set_ylabel("Validation Dice Score", fontsize=12)
    ax1.set_title("Stage 1 Screening: Dice vs. Wall-Clock Time", fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_ylim([0, 1.0])
    
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Dice Score", fontsize=12)
    ax2.set_title("Stage 1 Screening: Dice vs. Epochs", fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'stage1_screening_curves.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" 📈 Saved Stage 1 Screening plot to: {plot_path}")

def plot_final_comparison(results_summary, output_dir):
    """
    Plots final 5-fold OOF Mean Dice and Held-Out Test Dice comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = list(results_summary.keys())
    oof_means = [results_summary[m]["oof_mean_dice"] for m in models]
    test_means = [results_summary[m]["held_out_test_mean_dice"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, oof_means, width, label='5-Fold OOF Dice', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, test_means, width, label='20-Case Held-Out Test Dice', color='#d95f02')
    
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Final Benchmark Comparison: OOF vs Held-Out Test Set', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'final_model_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" 📊 Saved Final Model Comparison plot to: {plot_path}")
