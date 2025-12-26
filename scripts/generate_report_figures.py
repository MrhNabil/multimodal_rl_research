#!/usr/bin/env python
"""
COMPREHENSIVE DIAGRAM GENERATION FOR FINAL REPORT

Generates all scientific figures using matplotlib:
1. Architecture diagram (detailed flow)
2. All 61 experiments summary
3. Learning rate sensitivity analysis
4. Method comparison
5. Per-question-type accuracy
6. Reward function comparison
7. REINFORCE algorithm flowchart
8. Training pipeline visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines
import numpy as np
import json
import os

# Set professional academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.facecolor'] = 'white'

OUTPUT_DIR = r"d:\multimodal_rl_research\experiments\report_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD ALL EXPERIMENT DATA
# ============================================================

# Key experiment data (from COMPLETE_RESULTS_AUDIT.json)
EXPERIMENTS = {
    # Baselines
    'exp_001_frozen': {'method': 'frozen', 'acc': 0.2, 'category': 'baseline'},
    'exp_002_supervised': {'method': 'supervised', 'acc': 74.0, 'category': 'baseline'},
    'exp_003_rl_baseline': {'method': 'rl', 'acc': 47.6, 'category': 'baseline'},
    
    # Learning Rate Experiments (results_fixed - 1000 steps)
    'exp_004_lr_1e-5': {'method': 'rl', 'acc': 29.4, 'lr': 1e-5, 'category': 'lr_sweep'},
    'exp_005_lr_2e-5': {'method': 'rl', 'acc': 37.0, 'lr': 2e-5, 'category': 'lr_sweep'},
    'exp_006_lr_5e-5': {'method': 'rl', 'acc': 41.0, 'lr': 5e-5, 'category': 'lr_sweep'},
    'exp_007_lr_1e-4': {'method': 'rl', 'acc': 45.2, 'lr': 1e-4, 'category': 'lr_sweep'},
    'exp_008_lr_2e-4': {'method': 'rl', 'acc': 53.7, 'lr': 2e-4, 'category': 'lr_sweep'},
    'exp_009_lr_5e-4': {'method': 'rl', 'acc': 44.0, 'lr': 5e-4, 'category': 'lr_sweep'},
    'exp_010_lr_1e-3': {'method': 'rl', 'acc': 29.3, 'lr': 1e-3, 'category': 'lr_sweep'},
    'exp_011_lr_2e-3': {'method': 'rl', 'acc': 20.7, 'lr': 2e-3, 'category': 'lr_sweep'},
    'exp_012_lr_5e-3': {'method': 'rl', 'acc': 14.2, 'lr': 5e-3, 'category': 'lr_sweep'},
    'exp_013_lr_1e-2': {'method': 'rl', 'acc': 14.2, 'lr': 1e-2, 'category': 'lr_sweep'},
    
    # Reward Function Experiments
    'exp_014_reward_exact': {'method': 'rl', 'acc': 29.3, 'reward': 'exact', 'category': 'reward'},
    'exp_015_reward_partial': {'method': 'rl', 'acc': 29.3, 'reward': 'partial', 'category': 'reward'},
    'exp_016_reward_length': {'method': 'rl', 'acc': 32.4, 'reward': 'length_penalty', 'category': 'reward'},
    'exp_017_reward_combined': {'method': 'rl', 'acc': 32.4, 'reward': 'combined', 'category': 'reward'},
    'exp_018_reward_progressive': {'method': 'rl', 'acc': 32.4, 'reward': 'progressive', 'category': 'reward'},
    'exp_019_reward_exact_strict': {'method': 'rl', 'acc': 32.4, 'reward': 'exact_strict', 'category': 'reward'},
    'exp_020_reward_partial_strict': {'method': 'rl', 'acc': 32.4, 'reward': 'partial_strict', 'category': 'reward'},
    'exp_021_reward_combined_v2': {'method': 'rl', 'acc': 32.4, 'reward': 'combined_v2', 'category': 'reward'},
    'exp_022_reward_progressive_fast': {'method': 'rl', 'acc': 32.4, 'reward': 'prog_fast', 'category': 'reward'},
    'exp_023_reward_progressive_slow': {'method': 'rl', 'acc': 43.1, 'reward': 'prog_slow', 'category': 'reward'},
    
    # Question Type Experiments
    'exp_024_qtype_color': {'method': 'rl', 'acc': 43.1, 'qtype': 'color', 'category': 'qtype'},
    'exp_025_qtype_shape': {'method': 'rl', 'acc': 43.1, 'qtype': 'shape', 'category': 'qtype'},
    'exp_026_qtype_count': {'method': 'rl', 'acc': 43.1, 'qtype': 'count', 'category': 'qtype'},
    'exp_027_qtype_spatial': {'method': 'rl', 'acc': 43.1, 'qtype': 'spatial', 'category': 'qtype'},
    'exp_028_qtype_color_shape': {'method': 'rl', 'acc': 43.1, 'qtype': 'color+shape', 'category': 'qtype'},
    'exp_029_qtype_color_count': {'method': 'rl', 'acc': 43.1, 'qtype': 'color+count', 'category': 'qtype'},
}

# Per-type accuracy data
PER_TYPE_SUPERVISED = {'shape': 77.4, 'color': 75.7, 'count': 82.0, 'spatial': 61.3}
PER_TYPE_RL = {'shape': 71.8, 'color': 20.6, 'count': 58.0, 'spatial': 39.8}
PER_TYPE_HIGH_ACC = {'shape': 79.4, 'color': 71.3, 'count': 62.0, 'spatial': 62.1}


# ============================================================
# FIGURE 1: DETAILED ARCHITECTURE DIAGRAM
# ============================================================
def fig1_architecture():
    """Create detailed architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(8, 5.7, 'Multimodal VQA Architecture', fontsize=14, fontweight='bold', ha='center')
    ax.text(8, 5.3, 'CLIP (Frozen) + Trainable Projection + Question-Type-Specific Heads', 
            fontsize=10, ha='center', style='italic', color='gray')
    
    # Color scheme
    frozen_color = '#E3F2FD'  # Light blue for frozen
    train_color = '#FFF3E0'   # Light orange for trainable
    output_color = '#E8F5E9'  # Light green for output
    
    # INPUT: Image
    img_box = FancyBboxPatch((0.3, 2), 1.8, 2, boxstyle="round,pad=0.1",
                              facecolor='#ECEFF1', edgecolor='black', linewidth=2)
    ax.add_patch(img_box)
    ax.text(1.2, 3, 'Input\nImage', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1.2, 2.3, '224×224×3', ha='center', va='center', fontsize=8, color='gray')
    
    # CLIP ViT-B/32 (Frozen)
    clip_box = FancyBboxPatch((2.5, 1.5), 2.5, 3, boxstyle="round,pad=0.1",
                              facecolor=frozen_color, edgecolor='#1976D2', linewidth=2)
    ax.add_patch(clip_box)
    ax.text(3.75, 3.2, 'CLIP', ha='center', va='center', fontsize=11, fontweight='bold', color='#1976D2')
    ax.text(3.75, 2.7, 'ViT-B/32', ha='center', va='center', fontsize=9)
    ax.text(3.75, 2.2, '(FROZEN)', ha='center', va='center', fontsize=8, color='red', fontweight='bold')
    ax.text(3.75, 1.8, '151M params', ha='center', va='center', fontsize=7, color='gray')
    
    # Arrow
    ax.annotate('', xy=(2.4, 3), xytext=(2.1, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Visual Embedding
    emb_box = FancyBboxPatch((5.3, 2.2), 1.6, 1.6, boxstyle="round,pad=0.1",
                             facecolor='#E1F5FE', edgecolor='#0288D1', linewidth=2)
    ax.add_patch(emb_box)
    ax.text(6.1, 3, 'Visual\nEmbedding', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(6.1, 2.5, '512-d', ha='center', va='center', fontsize=8, color='gray')
    
    ax.annotate('', xy=(5.2, 3), xytext=(5, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Projection Layer (Trainable)
    proj_box = FancyBboxPatch((7.2, 2.2), 1.6, 1.6, boxstyle="round,pad=0.1",
                              facecolor=train_color, edgecolor='#F57C00', linewidth=2)
    ax.add_patch(proj_box)
    ax.text(8, 3, 'Projection', ha='center', va='center', fontsize=9, fontweight='bold', color='#E65100')
    ax.text(8, 2.5, 'MLP', ha='center', va='center', fontsize=8, color='gray')
    ax.text(8, 2.3, '(trainable)', ha='center', va='center', fontsize=7, color='gray')
    
    ax.annotate('', xy=(7.1, 3), xytext=(6.9, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Question Input (bottom)
    q_box = FancyBboxPatch((7.2, 0.5), 1.6, 1.2, boxstyle="round,pad=0.1",
                           facecolor='#FFF8E1', edgecolor='#FFA000', linewidth=2)
    ax.add_patch(q_box)
    ax.text(8, 1.1, 'Question', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(8, 0.7, '"What color?"', ha='center', va='center', fontsize=7, style='italic')
    
    ax.annotate('', xy=(8, 2.1), xytext=(8, 1.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Fusion Layer
    fusion_box = FancyBboxPatch((9.1, 2.2), 1.6, 1.6, boxstyle="round,pad=0.1",
                                facecolor=train_color, edgecolor='#F57C00', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(9.9, 3, 'Fusion', ha='center', va='center', fontsize=9, fontweight='bold', color='#E65100')
    ax.text(9.9, 2.5, 'Concat + MLP', ha='center', va='center', fontsize=7, color='gray')
    
    ax.annotate('', xy=(9, 3), xytext=(8.8, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Type-Specific Heads
    heads = [
        (11, 4.2, 'Color\nHead', '4 classes', '#FFCDD2'),
        (11, 3, 'Shape\nHead', '3 classes', '#C8E6C9'),
        (11, 1.8, 'Count\nHead', '4 classes', '#BBDEFB'),
        (11, 0.6, 'Spatial\nHead', '13 classes', '#F0F4C3'),
    ]
    
    for x, y, label, classes, color in heads:
        head_box = FancyBboxPatch((x, y-0.5), 1.4, 1, boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(head_box)
        ax.text(x+0.7, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows to heads
    for x, y, _, _, _ in heads:
        ax.annotate('', xy=(x-0.05, y), xytext=(10.7, 3),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1, connectionstyle='arc3,rad=0.1'))
    
    # Output Answers
    outputs = [
        (13, 4.2, 'red, blue,\ngreen, yellow'),
        (13, 3, 'cube, sphere,\ncylinder'),
        (13, 1.8, '0, 1, 2, 3'),
        (13, 0.6, 'red cube,\nblue sphere...'),
    ]
    
    for x, y, label in outputs:
        out_box = FancyBboxPatch((x, y-0.5), 2, 1, boxstyle="round,pad=0.05",
                                 facecolor=output_color, edgecolor='#388E3C', linewidth=1.5)
        ax.add_patch(out_box)
        ax.text(x+1, y, label, ha='center', va='center', fontsize=7)
    
    # Arrows to outputs
    for i, (x, y, _) in enumerate(outputs):
        ax.annotate('', xy=(x-0.05, y), xytext=(heads[i][0]+1.35, heads[i][1]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=frozen_color, edgecolor='#1976D2', label='Frozen (151M)'),
        mpatches.Patch(facecolor=train_color, edgecolor='#F57C00', label='Trainable (~1M)'),
        mpatches.Patch(facecolor=output_color, edgecolor='#388E3C', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_architecture.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_architecture.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Architecture Diagram")


# ============================================================
# FIGURE 2: ALL EXPERIMENTS OVERVIEW
# ============================================================
def fig2_all_experiments():
    """Bar chart showing all key experiments."""
    
    experiments = [
        ('Supervised (1000 steps)', 74.0, 'supervised'),
        ('HighAccuracyVQA', 68.7, 'supervised'),
        ('RL lr=2e-4 (best)', 53.7, 'rl'),
        ('RL baseline', 47.6, 'rl'),
        ('RL lr=1e-4', 45.2, 'rl'),
        ('RL lr=5e-4', 44.0, 'rl'),
        ('RL prog_slow reward', 43.1, 'rl'),
        ('RL lr=5e-5', 41.0, 'rl'),
        ('RL lr=2e-5', 37.0, 'rl'),
        ('Supervised (500 steps)', 33.7, 'supervised'),
        ('RL length_penalty', 32.4, 'rl'),
        ('RL lr=1e-3', 29.3, 'rl'),
        ('RL lr=1e-5', 29.4, 'rl'),
        ('RL lr=2e-3', 20.7, 'rl'),
        ('RL lr=5e-3', 14.2, 'rl'),
        ('Frozen baseline', 0.2, 'frozen'),
    ]
    
    names = [e[0] for e in experiments]
    accs = [e[1] for e in experiments]
    types = [e[2] for e in experiments]
    
    colors = {'rl': '#2196F3', 'supervised': '#4CAF50', 'frozen': '#F44336'}
    bar_colors = [colors[t] for t in types]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    
    bars = ax.barh(y_pos, accs, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Summary of All Experiments\n(Sorted by Accuracy)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 85)
    
    # Random baseline line
    ax.axvline(x=4.17, color='gray', linestyle='--', linewidth=1, label='Random (1/24)')
    
    # Legend
    patches = [
        mpatches.Patch(color='#4CAF50', label='Supervised'),
        mpatches.Patch(color='#2196F3', label='Reinforcement Learning'),
        mpatches.Patch(color='#F44336', label='Frozen Baseline'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=9)
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_all_experiments.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_all_experiments.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: All Experiments Overview")


# ============================================================
# FIGURE 3: LEARNING RATE SENSITIVITY
# ============================================================
def fig3_learning_rate():
    """Learning rate sensitivity analysis."""
    
    # Data from experiments
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    accuracies = [29.4, 37.0, 41.0, 45.2, 53.7, 44.0, 29.3, 20.7, 14.2, 14.2]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.semilogx(learning_rates, accuracies, 'o-', color='#2196F3', 
                linewidth=2, markersize=10, markeredgecolor='black')
    
    # Highlight optimal
    optimal_idx = np.argmax(accuracies)
    ax.scatter([learning_rates[optimal_idx]], [accuracies[optimal_idx]], 
               color='#F44336', s=200, zorder=5, edgecolor='black', linewidth=2)
    ax.annotate(f'Optimal: 2e-4\n({accuracies[optimal_idx]}%)', 
                xy=(learning_rates[optimal_idx], accuracies[optimal_idx]),
                xytext=(learning_rates[optimal_idx]*2, accuracies[optimal_idx]-8),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=2))
    
    # Add region annotations
    ax.axvspan(1e-5, 1e-4, alpha=0.1, color='blue', label='Underfitting region')
    ax.axvspan(1e-3, 1e-2, alpha=0.1, color='red', label='Overfitting region')
    ax.axvspan(1e-4, 1e-3, alpha=0.1, color='green', label='Optimal region')
    
    ax.set_xlabel('Learning Rate (log scale)', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Learning Rate Sensitivity Analysis\n(RL training, 1000-3000 steps)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_learning_rate.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Learning Rate Sensitivity")


# ============================================================
# FIGURE 4: METHOD COMPARISON (BAR CHART)
# ============================================================
def fig4_method_comparison():
    """Compare training methods."""
    
    methods = ['Frozen\nBaseline', 'Supervised\nLearning', 'Reinforcement\nLearning (best)']
    accuracies = [0.2, 74.0, 53.7]
    colors = ['#F44336', '#4CAF50', '#2196F3']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Training Method Comparison\n(1000 training steps, same dataset)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 85)
    
    # Random baseline
    ax.axhline(y=4.17, color='gray', linestyle='--', linewidth=1.5, label='Random (1/24 = 4.17%)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_method_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Method Comparison")


# ============================================================
# FIGURE 5: PER-QUESTION-TYPE ACCURACY
# ============================================================
def fig5_per_type():
    """Per question type accuracy comparison."""
    
    question_types = ['Color', 'Shape', 'Count', 'Spatial']
    
    supervised = [75.7, 77.4, 82.0, 61.3]
    rl = [20.6, 71.8, 58.0, 39.8]
    high_acc = [71.3, 79.4, 62.0, 62.1]
    
    x = np.arange(len(question_types))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, supervised, width, label='Supervised (74%)', 
                   color='#4CAF50', edgecolor='black')
    bars2 = ax.bar(x, rl, width, label='RL Baseline (47.6%)', 
                   color='#2196F3', edgecolor='black')
    bars3 = ax.bar(x + width, high_acc, width, label='HighAccuracyVQA (68.7%)', 
                   color='#9C27B0', edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy by Question Type\n(Comparing Training Methods)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(question_types, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate key findings
    ax.annotate('RL struggles\nwith color', xy=(0, 20.6), xytext=(0.3, 35),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_per_type.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_per_type.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Per-Question-Type Accuracy")


# ============================================================
# FIGURE 6: REINFORCE ALGORITHM FLOWCHART
# ============================================================
def fig6_reinforce_flowchart():
    """REINFORCE algorithm flowchart."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(6, 7.5, 'REINFORCE Policy Gradient Algorithm', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Boxes
    boxes = [
        (1, 5.5, 2.5, 1, 'Sample batch\n(images, questions)', '#E3F2FD'),
        (4.5, 5.5, 2.5, 1, 'Forward pass\nπ(a|s; θ)', '#E8F5E9'),
        (8, 5.5, 2.5, 1, 'Sample action\na ~ π(a|s)', '#FFF3E0'),
        (8, 3.5, 2.5, 1, 'Compute reward\nR = 1 if correct', '#FFEBEE'),
        (4.5, 3.5, 2.5, 1, 'Compute\nadvantage\nA = R - baseline', '#F3E5F5'),
        (1, 3.5, 2.5, 1, 'Policy gradient\n∇J = A·∇log π', '#E0F7FA'),
        (1, 1.5, 2.5, 1, 'Update weights\nθ = θ + α·∇J', '#FFFDE7'),
        (4.5, 1.5, 2.5, 1, 'Repeat until\nconvergence', '#E8EAF6'),
    ]
    
    for x, y, w, h, label, color in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9)
    
    # Arrows
    arrows = [
        ((3.5, 6), (4.4, 6)),
        ((7, 6), (7.9, 6)),
        ((9.25, 5.4), (9.25, 4.65)),
        ((7.9, 4), (7.05, 4)),
        ((4.4, 4), (3.55, 4)),
        ((2.25, 3.4), (2.25, 2.65)),
        ((3.5, 2), (4.4, 2)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Key equation box
    eq_box = FancyBboxPatch((7.5, 1), 4, 2, boxstyle="round,pad=0.1",
                            facecolor='#FAFAFA', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(eq_box)
    ax.text(9.5, 2.5, 'Key Equation:', fontsize=10, fontweight='bold', ha='center', color='#1976D2')
    ax.text(9.5, 1.8, r'$\nabla J(\theta) = \mathbb{E}[R \cdot \nabla_\theta \log \pi(a|s;\theta)]$', 
            fontsize=11, ha='center', style='italic')
    ax.text(9.5, 1.3, 'R=1 if correct, R=0 otherwise', fontsize=8, ha='center', color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_reinforce.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_reinforce.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: REINFORCE Flowchart")


# ============================================================
# FIGURE 7: REWARD FUNCTION COMPARISON
# ============================================================
def fig7_reward_comparison():
    """Compare different reward functions."""
    
    rewards = ['exact_match', 'partial', 'length_penalty', 'combined', 
               'progressive', 'prog_slow']
    accuracies = [29.3, 29.3, 32.4, 32.4, 32.4, 43.1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(rewards))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(rewards)))
    
    bars = ax.bar(x, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Highlight best
    bars[-1].set_color('#4CAF50')
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('_', '\n') for r in rewards], fontsize=9)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Reward Function Comparison\n(RL training with lr=2e-4)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 50)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_reward_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_reward_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 7: Reward Function Comparison")


# ============================================================
# FIGURE 8: EXPERIMENT CATEGORIES BREAKDOWN
# ============================================================
def fig8_categories():
    """Pie chart of experiment categories."""
    
    categories = ['Learning Rate\nExperiments', 'Reward Function\nExperiments', 
                  'Question Type\nExperiments', 'Baseline\nExperiments', 
                  'Architecture\nVariations']
    sizes = [10, 10, 6, 3, 4]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    explode = (0.05, 0.05, 0.05, 0.1, 0.05)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=categories, 
                                       colors=colors, autopct='%1.0f%%',
                                       shadow=True, startangle=90,
                                       wedgeprops=dict(edgecolor='black', linewidth=1))
    
    autotexts[0].set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(10)
    
    ax.set_title('Experiment Categories (Total: 33+ Experiments)', 
                 fontsize=12, fontweight='bold')
    
    # Add legend
    legend_text = [
        'LR: 1e-5 to 1e-2',
        'Reward: exact, partial, progressive...',
        'QType: color, shape, count, spatial',
        'Frozen, Supervised, RL baseline',
        'HighAccuracy, UltraHigh, Ensemble'
    ]
    ax.legend(wedges, legend_text, title="Details", loc="lower left", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_categories.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_categories.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 8: Experiment Categories")


# ============================================================
# RUN ALL
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING COMPREHENSIVE REPORT FIGURES")
    print("=" * 60)
    
    fig1_architecture()
    fig2_all_experiments()
    fig3_learning_rate()
    fig4_method_comparison()
    fig5_per_type()
    fig6_reinforce_flowchart()
    fig7_reward_comparison()
    fig8_categories()
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
