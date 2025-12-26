#!/usr/bin/env python
"""
Generate REAL plots from ACTUAL experimental data.
No fabricated values - all data comes from executed experiments.

Source verification:
- results_fixed: 7 experiments with 1000 training steps
- results_gpu: 29 experiments with 500 training steps
- high_accuracy: HighAccuracyVQA model results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import json

# Output directory
OUTPUT_DIR = r"d:\multimodal_rl_research\experiments\final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

# ============================================================
# DATA FROM ACTUAL EXPERIMENTS (results_fixed - 1000 steps)
# Source: experiments/results_fixed/batch_summary.json
# ============================================================

RESULTS_FIXED_1000_STEPS = {
    'exp_001_frozen': {'method': 'frozen', 'accuracy': 0.2, 'per_type': {'shape': 0.0, 'color': 0.0, 'count': 0.0, 'spatial': 0.8}},
    'exp_002_supervised': {'method': 'supervised', 'accuracy': 74.0, 'per_type': {'shape': 77.4, 'color': 75.7, 'count': 82.0, 'spatial': 61.3}},
    'exp_003_rl_baseline': {'method': 'rl', 'accuracy': 47.6, 'per_type': {'shape': 71.8, 'color': 20.6, 'count': 58.0, 'spatial': 39.8}},
    'exp_004_lr_1e-5': {'method': 'rl', 'accuracy': 29.4, 'per_type': {'shape': 0.0, 'color': 25.1, 'count': 58.0, 'spatial': 35.2}},
    'exp_005_lr_2e-5': {'method': 'rl', 'accuracy': 37.0, 'per_type': {'shape': 30.2, 'color': 25.1, 'count': 58.0, 'spatial': 35.2}},
    'exp_006_lr_5e-5': {'method': 'rl', 'accuracy': 41.0, 'per_type': {'shape': 44.0, 'color': 27.1, 'count': 58.0, 'spatial': 35.2}},
    'exp_007_lr_1e-4': {'method': 'rl', 'accuracy': 45.2, 'per_type': {'shape': 53.6, 'color': 28.3, 'count': 58.0, 'spatial': 41.0}},
}

# Data from results_gpu (500 steps)
RESULTS_GPU_500_STEPS = {
    'exp_008_lr_2e-4': {'method': 'rl', 'accuracy': 53.7, 'per_type': {'shape': 73.8, 'color': 63.2, 'count': 58.0, 'spatial': 20.7}},
    'exp_009_lr_5e-4': {'method': 'rl', 'accuracy': 44.0},
    'exp_010_lr_1e-3': {'method': 'rl', 'accuracy': 29.3},
    'exp_011_lr_2e-3': {'method': 'rl', 'accuracy': 20.7},
    'exp_012_lr_5e-3': {'method': 'rl', 'accuracy': 14.2},
    'exp_013_lr_1e-2': {'method': 'rl', 'accuracy': 14.2},
}

# High accuracy model result
HIGH_ACCURACY_MODEL = {
    'accuracy': 68.7,
    'method': 'supervised',
    'per_type': {'shape': 79.4, 'color': 71.3, 'count': 62.0, 'spatial': 62.1}
}

print("=" * 60)
print("GENERATING PLOTS FROM REAL EXPERIMENTAL DATA")
print("=" * 60)

# ============================================================
# FIGURE 1: Training Method Comparison (1000 steps)
# ============================================================
def fig1_method_comparison():
    """Bar chart: Frozen vs Supervised vs RL (from results_fixed, 1000 steps)"""
    
    methods = ['Frozen\nBaseline', 'Supervised\nLearning', 'Reinforcement\nLearning (best)']
    accuracies = [0.2, 74.0, 47.6]  # From results_fixed
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Training Method Comparison (1000 steps)\nSource: experiments/results_fixed/')
    ax.set_ylim(0, 85)
    ax.axhline(y=4.17, color='gray', linestyle='--', linewidth=1, label='Random (1/24 classes)')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_method_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Method Comparison (1000 steps)")

# ============================================================
# FIGURE 2: Learning Rate Sensitivity
# ============================================================
def fig2_learning_rate():
    """Learning rate vs accuracy (combined from results_fixed + results_gpu)"""
    
    # Combined data from both runs
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    
    # Accuracies from results_fixed (1000 steps) for lower LRs
    # and results_gpu (500 steps) for higher LRs
    accuracies_1000 = [29.4, 37.0, 41.0, 45.2, None, None, None, None, None, None]  # 1000 steps
    accuracies_500 = [None, None, None, None, 53.7, 44.0, 29.3, 20.7, 14.2, 14.2]  # 500 steps
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot 500 step results
    lrs_500 = [lr for lr, acc in zip(learning_rates, accuracies_500) if acc is not None]
    accs_500 = [acc for acc in accuracies_500 if acc is not None]
    ax.semilogx(lrs_500, accs_500, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='500 steps (results_gpu)')
    
    # Plot 1000 step results
    lrs_1000 = [lr for lr, acc in zip(learning_rates, accuracies_1000) if acc is not None]
    accs_1000 = [acc for acc in accuracies_1000 if acc is not None]
    ax.semilogx(lrs_1000, accs_1000, 's--', color='#ff7f0e', linewidth=2, markersize=8, label='1000 steps (results_fixed)')
    
    # Highlight optimal
    ax.scatter([2e-4], [53.7], color='red', s=150, zorder=5, edgecolor='black', linewidth=2)
    ax.annotate('Best: 2e-4\n(53.7%)', xy=(2e-4, 53.7), xytext=(4e-4, 45),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
    
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Learning Rate Sensitivity Analysis\nSource: experiments/results_fixed/ & results_gpu/')
    ax.set_ylim(0, 60)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_learning_rate.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_learning_rate.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Learning Rate Sensitivity")

# ============================================================
# FIGURE 3: Per-Question-Type Accuracy
# ============================================================
def fig3_per_type():
    """Per question type accuracy comparison"""
    
    question_types = ['Shape', 'Color', 'Count', 'Spatial']
    
    # From results_fixed exp_002 (supervised, 1000 steps)
    supervised_1000 = [77.4, 75.7, 82.0, 61.3]
    
    # From results_fixed exp_003 (RL, 1000 steps)
    rl_1000 = [71.8, 20.6, 58.0, 39.8]
    
    # From high_accuracy_model
    high_acc = [79.4, 71.3, 62.0, 62.1]
    
    x = np.arange(len(question_types))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, supervised_1000, width, label='Supervised (1000 steps)', color='#ff7f0e', edgecolor='black')
    bars2 = ax.bar(x, rl_1000, width, label='RL (1000 steps)', color='#2ca02c', edgecolor='black')
    bars3 = ax.bar(x + width, high_acc, width, label='HighAccuracyVQA', color='#9467bd', edgecolor='black')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Question-Type Accuracy\nSource: experiments/results_fixed/ & high_accuracy/')
    ax.set_xticks(x)
    ax.set_xticklabels(question_types)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_per_type.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_per_type.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Per-Question-Type Accuracy")

# ============================================================
# FIGURE 4: Complete Results Summary
# ============================================================
def fig4_summary():
    """Summary bar chart of all key experiments"""
    
    experiments = [
        ('Supervised\n(1000 steps)', 74.0, 'supervised'),
        ('HighAccuracyVQA', 68.7, 'supervised'),
        ('RL lr=2e-4\n(500 steps)', 53.7, 'rl'),
        ('RL baseline\n(1000 steps)', 47.6, 'rl'),
        ('RL lr=1e-4\n(1000 steps)', 45.2, 'rl'),
        ('RL lr=5e-5\n(1000 steps)', 41.0, 'rl'),
        ('RL lr=2e-5\n(1000 steps)', 37.0, 'rl'),
        ('Frozen', 0.2, 'frozen'),
    ]
    
    names = [e[0] for e in experiments]
    accs = [e[1] for e in experiments]
    types = [e[2] for e in experiments]
    
    colors = {'rl': '#2ca02c', 'supervised': '#ff7f0e', 'frozen': '#d62728'}
    bar_colors = [colors[t] for t in types]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accs, color=bar_colors, edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2, f'{acc:.1f}%', 
                va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Complete Experiment Summary\nSource: All experiments from results_fixed/, results_gpu/, high_accuracy/')
    ax.set_xlim(0, 85)
    
    # Legend
    patches = [mpatches.Patch(color=c, label=l.capitalize()) for l, c in colors.items()]
    ax.legend(handles=patches, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_summary.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_summary.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Complete Summary")

# ============================================================
# FIGURE 5: Architecture Diagram (Code-generated, not AI art)
# ============================================================
def fig5_architecture():
    """Simple architecture diagram using matplotlib shapes"""
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Component boxes
    components = [
        (0.3, 1.2, 1.8, 1.6, 'Image\n(224×224)', '#e6f2ff'),
        (2.5, 1.2, 1.8, 1.6, 'CLIP\nViT-B/32\n(Frozen)', '#b3d9ff'),
        (4.7, 1.2, 1.8, 1.6, 'Visual\nEmbedding\n(512-d)', '#80c1ff'),
        (6.9, 1.2, 1.8, 1.6, 'Projection\n(Trainable)', '#4da8ff'),
        (9.1, 1.2, 1.8, 1.6, 'Fusion +\nClassifier', '#1a8cff'),
        (11.5, 1.2, 1.8, 1.6, 'Answer\n(24 classes)', '#0066cc'),
    ]
    
    for x, y, w, h, label, color in components:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        text_color = 'white' if color in ['#1a8cff', '#0066cc'] else 'black'
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, 
                fontweight='bold', color=text_color)
    
    # Question input (bottom)
    q_box = mpatches.FancyBboxPatch((6.9, 0), 1.8, 0.9, boxstyle="round,pad=0.05",
                                     facecolor='#ffe6cc', edgecolor='black', linewidth=1.5)
    ax.add_patch(q_box)
    ax.text(7.8, 0.45, 'Question', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=2)
    ax.annotate('', xy=(2.4, 2), xytext=(2.1, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(4.6, 2), xytext=(4.3, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(6.8, 2), xytext=(6.5, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(9.0, 2), xytext=(8.7, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(11.4, 2), xytext=(11.1, 2), arrowprops=arrow_style)
    ax.annotate('', xy=(7.8, 1.2), xytext=(7.8, 0.9), arrowprops=arrow_style)
    
    ax.set_title('Model Architecture: Frozen CLIP + Trainable Projection + Classifier\n(Code-generated diagram, not AI art)', 
                 fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_architecture.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_architecture.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Architecture Diagram")

# ============================================================
# RUN ALL
# ============================================================
if __name__ == "__main__":
    fig1_method_comparison()
    fig2_learning_rate()
    fig3_per_type()
    fig4_summary()
    fig5_architecture()
    
    print("\n" + "=" * 60)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("Both PNG and PDF formats available")
    print("=" * 60)
    print("\nDATA SOURCES (for verification):")
    print("- results_fixed/batch_summary.json (7 exps, 1000 steps)")
    print("- results_gpu/batch_summary.json (29 exps, 500 steps)")
    print("- high_accuracy/results.json (HighAccuracyVQA model)")
