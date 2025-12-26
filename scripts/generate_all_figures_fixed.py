#!/usr/bin/env python
"""
ALL FIGURES FIXED - NO UNICODE SYMBOLS

Issues fixed:
- Replaced nabla/theta/pi with plain text
- Fixed "not equal" symbol 
- Made all text ASCII-safe
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

OUTPUT_DIR = r"d:\multimodal_rl_research\experiments\report_figures_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig01_architecture():
    """Clean architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_facecolor('white')
    
    ax.text(7, 3.7, 'Model Architecture: CLIP + Trainable MLP + Classification Heads', 
            fontsize=13, fontweight='bold', ha='center')
    
    boxes = [
        (0.3, 1, 2, 1.8, 'INPUT\n224x224 Image', '#ECEFF1', 'black'),
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
    
    arrow_x = [2.3, 5.0, 7.3, 9.6, 11.7]
    for x in arrow_x:
        ax.annotate('', xy=(x+0.4, 1.9), xytext=(x, 1.9),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    q_box = FancyBboxPatch((7.8, 0.1), 1.8, 0.7, boxstyle="round,pad=0.05",
                           facecolor='#FFF8E1', edgecolor='#F9A825', linewidth=2)
    ax.add_patch(q_box)
    ax.text(8.7, 0.45, 'Question Type', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(8.7, 0.95), xytext=(8.7, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    legend_elements = [
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='#1565C0', label='Frozen (151M)'),
        mpatches.Patch(facecolor='#FFE0B2', edgecolor='#E65100', label='Trainable (~1M)'),
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9, framealpha=0.9)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig01_architecture.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig01")


def fig02_training_pipeline():
    """Clean training pipeline."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    ax.text(6, 2.7, 'Training Pipeline', fontsize=13, fontweight='bold', ha='center')
    
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
    
    for i in range(len(stages)-1):
        ax.annotate('', xy=(stages[i+1][0]-0.1, 1.4), xytext=(stages[i][0]+1.5, 1.4),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.annotate('Repeat', xy=(1.5, 0.5), xytext=(10.5, 0.5),
                fontsize=10, ha='right', color='#D32F2F',
                arrowprops=dict(arrowstyle='<-', color='#D32F2F', lw=2,
                               connectionstyle='arc3,rad=-0.3'))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig02_training_pipeline.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig02")


def fig03_method_comparison():
    """Clean bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = ['Frozen', 'Supervised', 'RL (best)']
    accuracies = [0.2, 74.0, 53.7]
    colors = ['#F44336', '#4CAF50', '#2196F3']
    
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=2, width=0.5)
    
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
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig03_method_comparison.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig03")


def fig04_learning_rate():
    """Learning rate sensitivity."""
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    accuracies = [29.4, 37.0, 41.0, 45.2, 53.7, 44.0, 29.3, 20.7, 14.2, 14.2]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.axvspan(1e-5, 1e-4, alpha=0.15, color='blue', label='Underfitting')
    ax.axvspan(1e-4, 1e-3, alpha=0.15, color='green', label='Optimal')
    ax.axvspan(1e-3, 1e-2, alpha=0.15, color='red', label='Overfitting')
    
    ax.semilogx(learning_rates, accuracies, 'o-', color='#1976D2', 
                linewidth=2.5, markersize=12, markeredgecolor='black', markerfacecolor='white')
    
    for lr, acc in zip(learning_rates, accuracies):
        ax.annotate(f'{acc:.1f}%', xy=(lr, acc), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=8)
    
    optimal_idx = np.argmax(accuracies)
    ax.scatter([learning_rates[optimal_idx]], [accuracies[optimal_idx]], 
               color='#F44336', s=300, zorder=5, edgecolor='black', linewidth=3, marker='*')
    
    ax.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Learning Rate Sensitivity Analysis', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    ax.annotate('OPTIMAL\nlr = 2e-4\n53.7%', 
                xy=(2e-4, 53.7), xytext=(5e-4, 58),
                fontsize=11, fontweight='bold', color='#D32F2F',
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig04_learning_rate.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig04")


def fig05_per_type():
    """Per-type accuracy."""
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
    
    ax.annotate('RL struggles!', xy=(0.175, 20.6), xytext=(0.5, 40),
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig05_per_type.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig05")


def fig06_all_experiments():
    """All experiments bar chart."""
    experiments = [
        ('Supervised (1000)', 74.0, 'sup'),
        ('HighAccuracyVQA', 68.7, 'sup'),
        ('RL lr=2e-4', 53.7, 'rl'),
        ('RL baseline', 47.6, 'rl'),
        ('RL lr=1e-4', 45.2, 'rl'),
        ('RL lr=5e-4', 44.0, 'rl'),
        ('RL prog_slow', 43.1, 'rl'),
        ('RL lr=5e-5', 41.0, 'rl'),
        ('RL lr=2e-5', 37.0, 'rl'),
        ('Supervised 500', 33.7, 'sup'),
        ('RL combined', 32.4, 'rl'),
        ('RL lr=1e-3', 29.3, 'rl'),
        ('RL lr=1e-5', 29.4, 'rl'),
        ('RL lr=2e-3', 20.7, 'rl'),
        ('RL lr=5e-3', 14.2, 'rl'),
        ('Frozen', 0.2, 'frozen'),
    ]
    
    names = [e[0] for e in experiments]
    accs = [e[1] for e in experiments]
    types = [e[2] for e in experiments]
    
    colors = {'rl': '#2196F3', 'sup': '#4CAF50', 'frozen': '#F44336'}
    bar_colors = [colors[t] for t in types]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(names))
    
    bars = ax.barh(y_pos, accs, color=bar_colors, edgecolor='black', linewidth=0.8, height=0.7)
    
    for bar, acc in zip(bars, accs):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('All Key Experiments Summary', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 85)
    
    patches = [
        mpatches.Patch(color='#4CAF50', label='Supervised'),
        mpatches.Patch(color='#2196F3', label='RL'),
        mpatches.Patch(color='#F44336', label='Frozen'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig06_all_experiments.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig06")


def fig07_reinforce():
    """REINFORCE - FIXED: No special Unicode chars."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(6, 7.5, 'REINFORCE Policy Gradient Algorithm', fontsize=14, fontweight='bold', ha='center')
    
    # Steps using plain ASCII
    steps = [
        (1, 5.5, 3, 1, '1. Sample batch (image, question, answer)', '#E3F2FD'),
        (1, 4, 3, 1, '2. Forward pass: policy(a|s)', '#E8F5E9'),
        (1, 2.5, 3, 1, '3. Sample action from policy', '#FFF3E0'),
        (5.5, 5.5, 3, 1, '4. Compute reward (R=1 if correct)', '#FFEBEE'),
        (5.5, 4, 3, 1, '5. Advantage: A = R - baseline', '#F3E5F5'),
        (5.5, 2.5, 3, 1, '6. Policy gradient: dJ = A * d(log policy)', '#E0F7FA'),
        (5.5, 1, 3, 1, '7. Update: weights += lr * dJ', '#FFFDE7'),
    ]
    
    for x, y, w, h, label, color in steps:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9)
    
    # Arrows
    for i in range(3):
        ax.annotate('', xy=(2.5, 4+i*1.5-0.5), xytext=(2.5, 4+i*1.5+0.5-0.1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(5.4, 5.5), xytext=(4.1, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    for i in range(3):
        ax.annotate('', xy=(7, 2.5+i*1.5-0.5), xytext=(7, 2.5+i*1.5+0.5-0.1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Key equation box - PLAIN TEXT ONLY
    eq_box = FancyBboxPatch((9, 3), 2.8, 3.5, boxstyle="round,pad=0.15",
                            facecolor='#FAFAFA', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(eq_box)
    ax.text(10.4, 5.8, 'Key Equation', fontsize=11, fontweight='bold', ha='center', color='#1976D2')
    ax.text(10.4, 5.0, 'Gradient J =', fontsize=10, ha='center')
    ax.text(10.4, 4.4, 'E[R * grad log policy]', fontsize=10, ha='center')
    ax.text(10.4, 3.6, 'R = 1 (correct)', fontsize=9, ha='center', color='green')
    ax.text(10.4, 3.2, 'R = 0 (wrong)', fontsize=9, ha='center', color='red')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig07_reinforce.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig07")


def fig08_reward_functions():
    """Reward function comparison."""
    rewards = ['Exact\nMatch', 'Partial\nMatch', 'Length\nPenalty', 'Combined', 'Progressive\n(slow)']
    accuracies = [29.3, 29.3, 32.4, 32.4, 43.1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#90CAF9', '#90CAF9', '#64B5F6', '#42A5F5', '#4CAF50']
    
    bars = ax.bar(rewards, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Reward Function Comparison\n(RL training, lr=2e-4)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 55)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.annotate('Best reward\nfunction', xy=(4, 43.1), xytext=(4, 50),
                fontsize=10, ha='center', color='#2E7D32', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig08_reward_functions.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig08")


def fig09_dataset():
    """Dataset statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    splits = ['Train\n(5000)', 'Val\n(1000)', 'Test\n(1000)']
    sizes = [5000, 1000, 1000]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    ax1.pie(sizes, labels=splits, colors=colors, autopct='%1.0f%%',
            shadow=True, startangle=90, explode=(0.05, 0.05, 0.05),
            wedgeprops=dict(edgecolor='black', linewidth=1))
    ax1.set_title('Dataset Split', fontsize=12, fontweight='bold')
    
    ax2 = axes[1]
    qtypes = ['Color', 'Shape', 'Count', 'Spatial']
    counts = [25, 25, 25, 25]
    colors2 = ['#FFCDD2', '#C8E6C9', '#BBDEFB', '#FFF9C4']
    
    bars = ax2.barh(qtypes, counts, color=colors2, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Percentage (%)', fontsize=11)
    ax2.set_title('Question Type Distribution\n(Balanced)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 35)
    
    for bar in bars:
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 '25%', va='center', fontsize=10, fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig09_dataset.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig09")


def fig10_key_findings():
    """Key findings - FIXED: No special chars."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(6, 7.3, 'Key Research Findings', fontsize=16, fontweight='bold', ha='center')
    
    # FIXED: Using plain text instead of symbols
    findings = [
        (0.5, 5.0, '1', 'Supervised > RL', '74% vs 53.7%', '#66BB6A'),
        (4.25, 5.0, '2', 'Optimal LR', '2e-4', '#42A5F5'),
        (8, 5.0, '3', 'Color vs Shape', '20% vs 72%', '#FFA726'),
        (0.5, 2.0, '4', 'Frozen Works', '0.2% to 74%', '#AB47BC'),
        (4.25, 2.0, '5', 'Spatial Hard', '24-61%', '#EF5350'),
        (8, 2.0, '6', 'More Data No Help', '50K to 61%', '#8D6E63'),
    ]
    
    for x, y, num, title, detail, color in findings:
        box = FancyBboxPatch((x, y), 3, 2, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
        ax.add_patch(box)
        
        # Number circle
        from matplotlib.patches import Circle
        circle = Circle((x+0.4, y+1.6), 0.25, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x+0.4, y+1.6, num, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Text
        ax.text(x+1.5, y+1.4, title, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(x+1.5, y+0.7, detail, ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_key_findings.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig10")


if __name__ == "__main__":
    print("=" * 50)
    print("REGENERATING ALL FIGURES (ASCII ONLY)")
    print("=" * 50)
    
    fig01_architecture()
    fig02_training_pipeline()
    fig03_method_comparison()
    fig04_learning_rate()
    fig05_per_type()
    fig06_all_experiments()
    fig07_reinforce()
    fig08_reward_functions()
    fig09_dataset()
    fig10_key_findings()
    
    print("\n" + "=" * 50)
    print("ALL FIGURES REGENERATED")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 50)
