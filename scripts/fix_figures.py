#!/usr/bin/env python
"""
FIXED FIGURES - Clean and Simple

Fixing issues with figures 1, 2, 3, 5
Using simpler layouts with proper spacing
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'

OUTPUT_DIR = r"d:\multimodal_rl_research\experiments\report_figures_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig01_architecture():
    """FIXED: Simple horizontal architecture diagram."""
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Title
    ax.text(7, 3.7, 'Model Architecture: CLIP + Trainable MLP + Classification Heads', 
            fontsize=13, fontweight='bold', ha='center')
    
    # Simple boxes in a row
    boxes = [
        (0.3, 1, 2, 1.8, 'INPUT\n224×224 Image', '#ECEFF1', 'black'),
        (2.8, 1, 2.2, 1.8, 'CLIP ViT-B/32\n(FROZEN)\n151M params', '#BBDEFB', '#1565C0'),
        (5.5, 1, 1.8, 1.8, 'Visual\nEmbedding\n512-d', '#E1F5FE', '#0277BD'),
        (7.8, 1, 1.8, 1.8, 'Projection\nMLP\n(trainable)', '#FFE0B2', '#E65100'),
        (10.1, 1, 1.6, 1.8, 'Fusion\nLayer', '#FFE0B2', '#E65100'),
        (12.2, 1, 1.5, 1.8, 'Output\nHeads\n(4 types)', '#C8E6C9', '#2E7D32'),
    ]
    
    for x, y, w, h, label, facecolor, edgecolor in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows between boxes
    arrow_x = [2.3, 5.0, 7.3, 9.6, 11.7]
    for x in arrow_x:
        ax.annotate('', xy=(x+0.4, 1.9), xytext=(x, 1.9),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Question input below
    q_box = FancyBboxPatch((7.8, 0.1), 1.8, 0.7, boxstyle="round,pad=0.05",
                           facecolor='#FFF8E1', edgecolor='#F9A825', linewidth=2)
    ax.add_patch(q_box)
    ax.text(8.7, 0.45, 'Question Type', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrow from question to fusion
    ax.annotate('', xy=(8.7, 0.95), xytext=(8.7, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='#1565C0', label='Frozen (151M)'),
        mpatches.Patch(facecolor='#FFE0B2', edgecolor='#E65100', label='Trainable (~1M)'),
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig01_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Architecture (FIXED - horizontal layout)")


def fig02_training_pipeline():
    """FIXED: Clean training pipeline."""
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    ax.text(6, 2.7, 'Training Pipeline', fontsize=13, fontweight='bold', ha='center')
    
    # Simple horizontal boxes
    stages = [
        (0.2, 0.8, 'Load\nData', '#E3F2FD'),
        (1.9, 0.8, 'CLIP\nEncode', '#BBDEFB'),
        (3.6, 0.8, 'Forward\nPass', '#90CAF9'),
        (5.3, 0.8, 'Compute\nLoss', '#64B5F6'),
        (7.0, 0.8, 'Backward\nPass', '#42A5F5'),
        (8.7, 0.8, 'Update\nWeights', '#2196F3'),
        (10.4, 0.8, 'Evaluate', '#1976D2'),
    ]
    
    for i, (x, y, label, color) in enumerate(stages):
        box = FancyBboxPatch((x, y), 1.4, 1.2, boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.7, y + 0.6, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white' if i > 3 else 'black')
    
    # Arrows
    for i in range(len(stages)-1):
        ax.annotate('', xy=(stages[i+1][0]-0.1, 1.4), xytext=(stages[i][0]+1.5, 1.4),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Loop arrow at bottom
    ax.annotate('Repeat', xy=(1.5, 0.5), xytext=(10.5, 0.5),
                fontsize=10, ha='right', color='#D32F2F',
                arrowprops=dict(arrowstyle='<-', color='#D32F2F', lw=2,
                               connectionstyle='arc3,rad=-0.3'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig02_training_pipeline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Training Pipeline (FIXED)")


def fig03_method_comparison():
    """FIXED: Clean method comparison bar chart."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = ['Frozen', 'Supervised', 'RL (best)']
    accuracies = [0.2, 74.0, 53.7]
    colors = ['#F44336', '#4CAF50', '#2196F3']
    
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=2, width=0.5)
    
    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Training Method Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 85)
    ax.axhline(y=4.17, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(2.5, 6, 'Random (4.17%)', fontsize=9, color='gray')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig03_method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Method Comparison (FIXED)")


def fig05_per_type():
    """FIXED: Clean per-type accuracy comparison."""
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    types = ['Color', 'Shape', 'Count', 'Spatial']
    supervised = [75.7, 77.4, 82.0, 61.3]
    rl = [20.6, 71.8, 58.0, 39.8]
    
    x = np.arange(len(types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, supervised, width, label='Supervised', 
                   color='#4CAF50', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, rl, width, label='RL', 
                   color='#2196F3', edgecolor='black', linewidth=1)
    
    # Add values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Question Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Highlight low value
    ax.annotate('RL struggles!', xy=(0.175, 20.6), xytext=(0.5, 40),
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig05_per_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Per-Type Accuracy (FIXED)")


if __name__ == "__main__":
    print("=" * 50)
    print("FIXING FIGURES 1, 2, 3, 5")
    print("=" * 50)
    
    fig01_architecture()
    fig02_training_pipeline()
    fig03_method_comparison()
    fig05_per_type()
    
    print("\n" + "=" * 50)
    print("FIXED FIGURES SAVED")
    print("=" * 50)
