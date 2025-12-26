#!/usr/bin/env python
"""Generate Real Scientific Figures from Experimental Data.

These are proper matplotlib figures from actual data - NOT AI-generated images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import json

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = r"d:\multimodal_rl_research\experiments\real_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FIGURE 1: Training Method Comparison (Bar Chart)
# ============================================================
def fig1_method_comparison():
    methods = ['Frozen\nBaseline', 'Supervised\nLearning', 'Reinforcement\nLearning']
    accuracies = [13.2, 33.7, 53.7]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Comparison of Training Methods on VQA Task')
    ax.set_ylim(0, 65)
    ax.axhline(y=4.17, color='gray', linestyle='--', linewidth=1, label='Random Guess (1/24)')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_method_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Method Comparison")

# ============================================================
# FIGURE 2: Learning Rate Sensitivity (Line Plot)
# ============================================================
def fig2_learning_rate():
    # Actual experimental data
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    accuracies = [18.8, 14.2, 32.4, 31.7, 53.7, 44.0, 29.3, 20.7, 14.2, 14.2]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(learning_rates, accuracies, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    
    # Highlight optimal
    optimal_idx = accuracies.index(max(accuracies))
    ax.scatter([learning_rates[optimal_idx]], [accuracies[optimal_idx]], 
               color='red', s=150, zorder=5, edgecolor='black', linewidth=2)
    ax.annotate(f'Optimal: {learning_rates[optimal_idx]}\n({max(accuracies):.1f}%)', 
                xy=(learning_rates[optimal_idx], max(accuracies)),
                xytext=(learning_rates[optimal_idx]*3, max(accuracies)-5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
    
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Effect of Learning Rate on RL Training Performance')
    ax.set_ylim(0, 60)
    ax.axhline(y=13.2, color='gray', linestyle='--', linewidth=1, label='Frozen Baseline')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_learning_rate.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_learning_rate.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Learning Rate Sensitivity")

# ============================================================
# FIGURE 3: Reward Function Comparison
# ============================================================
def fig3_reward_functions():
    rewards = ['Exact\nMatch', 'Partial\nMatch', 'Length\nPenalty', 'Combined', 'Progressive']
    accuracies = [29.3, 29.3, 32.4, 32.4, 43.1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(rewards)))
    bars = ax.bar(rewards, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Comparison of Reward Shaping Strategies')
    ax.set_ylim(0, 55)
    ax.axhline(y=53.7, color='green', linestyle='--', linewidth=1.5, label='Best RL (lr=2e-4)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_reward_functions.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_reward_functions.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Reward Functions")

# ============================================================
# FIGURE 4: All Experiments Summary (Horizontal Bar)
# ============================================================
def fig4_all_experiments():
    # All 29 experiments
    experiments = [
        ('exp_008_lr_2e-4', 53.7, 'rl'),
        ('exp_009_lr_5e-4', 44.0, 'rl'),
        ('exp_023_progressive_slow', 43.1, 'rl'),
        ('exp_002_supervised', 33.7, 'supervised'),
        ('exp_006_lr_5e-5', 32.4, 'rl'),
        ('exp_003_rl_baseline', 31.7, 'rl'),
        ('exp_010_lr_1e-3', 29.3, 'rl'),
        ('exp_011_lr_2e-3', 20.7, 'rl'),
        ('exp_004_lr_1e-5', 18.8, 'rl'),
        ('exp_005_lr_2e-5', 14.2, 'rl'),
        ('exp_001_frozen', 13.2, 'frozen'),
    ]
    
    names = [e[0].replace('exp_', '').replace('_', ' ') for e in experiments]
    accs = [e[1] for e in experiments]
    types = [e[2] for e in experiments]
    
    colors = {'rl': '#2ca02c', 'supervised': '#ff7f0e', 'frozen': '#d62728'}
    bar_colors = [colors[t] for t in types]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accs, color=bar_colors, edgecolor='black', linewidth=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Summary of Key Experiments')
    ax.set_xlim(0, 60)
    
    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    ax.legend(handles=patches, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_experiment_summary.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_experiment_summary.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Experiment Summary")

# ============================================================
# FIGURE 5: Architecture Diagram (Simple, Code-Generated)
# ============================================================
def fig5_architecture():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Boxes
    boxes = [
        (0.5, 1.5, 1.5, 1.5, 'Image\nInput', '#e6f3ff'),
        (2.5, 1.5, 1.5, 1.5, 'CLIP\n(Frozen)', '#b3d9ff'),
        (4.5, 1.5, 1.5, 1.5, 'Projection\nLayer', '#80bfff'),
        (6.5, 0.5, 1.5, 1.5, 'Fusion', '#4da6ff'),
        (6.5, 2.5, 1.5, 1.5, 'Question\nEncoder', '#cce6ff'),
        (8.5, 1.5, 1.5, 1.5, 'MLP\nClassifier', '#1a8cff'),
        (10.5, 1.5, 1.5, 1.5, 'Answer', '#0066cc'),
    ]
    
    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [(2, 2.25), (3.5, 2.25), (5.5, 2.25), (6, 1.25), (8, 2.25), (10, 2.25)]
    for i in range(len(arrows)-1):
        ax.annotate('', xy=(arrows[i+1][0]+0.5, arrows[i+1][1]), 
                   xytext=(arrows[i][0]+0.5, arrows[i][1]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Question input arrow
    ax.annotate('', xy=(6.5, 3.25), xytext=(5.5, 3.25),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(4.5, 3.25, 'Question\nInput', ha='center', va='center', fontsize=10)
    
    # Fusion arrows
    ax.annotate('', xy=(7.25, 2.5), xytext=(7.25, 2.0),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_title('Model Architecture: Frozen CLIP + Trainable Projection + Classifier', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_architecture.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_architecture.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Architecture Diagram")

# ============================================================
# FIGURE 6: REINFORCE Algorithm Flowchart
# ============================================================
def fig6_reinforce_algorithm():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Flowchart boxes
    boxes = [
        (3, 5, 2, 0.7, 'Sample (s, q) pair', '#e6f3ff'),
        (3, 4, 2, 0.7, 'Forward pass:\nπ(a|s,q)', '#b3d9ff'),
        (3, 3, 2, 0.7, 'Sample action a\nfrom π', '#80bfff'),
        (3, 2, 2, 0.7, 'Get reward:\nR = 1 if correct', '#4da6ff'),
        (3, 1, 2, 0.7, 'Update:\n∇θ = R·∇log π', '#1a8cff'),
    ]
    
    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9)
    
    # Arrows
    for i in range(4):
        ax.annotate('', xy=(4, 4.3 - i), xytext=(4, 5 - i),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Loop back arrow
    ax.annotate('', xy=(5.2, 5.35), xytext=(5.2, 1.35),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, 
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(6, 3, 'Repeat for\neach batch', fontsize=9, ha='left')
    
    ax.set_title('REINFORCE Training Algorithm', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_reinforce_flowchart.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_reinforce_flowchart.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: REINFORCE Flowchart")

# ============================================================
# RUN ALL
# ============================================================
if __name__ == "__main__":
    print("Generating real scientific figures from experimental data...\n")
    fig1_method_comparison()
    fig2_learning_rate()
    fig3_reward_functions()
    fig4_all_experiments()
    fig5_architecture()
    fig6_reinforce_algorithm()
    print(f"\n✓ All figures saved to: {OUTPUT_DIR}")
    print("✓ Both PNG and PDF formats available")
