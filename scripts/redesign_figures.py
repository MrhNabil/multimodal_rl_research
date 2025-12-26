#!/usr/bin/env python
"""
COMPLETELY REDESIGNED FIGURES

Fixed issues:
- Fig 1: Legend moved away from boxes, larger spacing
- Fig 5 (REINFORCE): Larger boxes, text fits inside, proper arrows
- Fig 8 (Reward): Annotation moved up to avoid overlap
- Fig 10: Number circles moved inside boxes, not overlapping with title
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = r"d:\multimodal_rl_research\experiments\report_figures_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig01_architecture():
    """Architecture - completely redesigned with no overlaps."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Title at top with more space
    ax.text(7, 4.7, 'Model Architecture', fontsize=14, fontweight='bold', ha='center')
    
    # Main boxes - higher up to leave room for legend
    y_main = 2.2
    box_height = 1.6
    
    boxes = [
        (0.2, y_main, 1.8, box_height, 'INPUT\n224x224', '#ECEFF1', 'black'),
        (2.3, y_main, 2.0, box_height, 'CLIP\nViT-B/32\n(FROZEN)', '#BBDEFB', '#1565C0'),
        (4.6, y_main, 1.6, box_height, 'Embed\n512-d', '#E1F5FE', '#0277BD'),
        (6.5, y_main, 1.6, box_height, 'Project\nMLP', '#FFE0B2', '#E65100'),
        (8.4, y_main, 1.6, box_height, 'Fusion', '#FFE0B2', '#E65100'),
        (10.3, y_main, 1.6, box_height, 'Classify', '#FFE0B2', '#E65100'),
        (12.2, y_main, 1.5, box_height, 'Output\n4 types', '#C8E6C9', '#2E7D32'),
    ]
    
    for x, y, w, h, label, facecolor, edgecolor in boxes:
        box = Rectangle((x, y), w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrow_positions = [(2.0, 3.0), (4.3, 3.0), (6.2, 3.0), (8.1, 3.0), (10.0, 3.0), (11.9, 3.0)]
    for x, y in arrow_positions:
        ax.annotate('', xy=(x+0.25, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Question input - separate area
    q_box = Rectangle((6.5, 0.8), 1.6, 0.9, facecolor='#FFF8E1', edgecolor='#F9A825', linewidth=2)
    ax.add_patch(q_box)
    ax.text(7.3, 1.25, 'Question', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(7.3, 2.15), xytext=(7.3, 1.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Legend - completely separate at very bottom
    ax.text(3.5, 0.2, 'Frozen (151M)', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='#BBDEFB', edgecolor='#1565C0'))
    ax.text(7, 0.2, 'Trainable (~1M)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFE0B2', edgecolor='#E65100'))
    ax.text(10.5, 0.2, 'Output', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#C8E6C9', edgecolor='#2E7D32'))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig01_architecture.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig01 - Architecture (redesigned)")


def fig05_reinforce():
    """REINFORCE - completely redesigned with larger boxes."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'REINFORCE Algorithm', fontsize=14, fontweight='bold', ha='center')
    
    # LEFT COLUMN - Steps 1-3
    left_steps = [
        (0.5, 7, 4, 1.2, 'Step 1: Sample Batch\n(image, question, answer)'),
        (0.5, 5, 4, 1.2, 'Step 2: Forward Pass\nCompute policy probabilities'),
        (0.5, 3, 4, 1.2, 'Step 3: Sample Action\nPick answer from policy'),
    ]
    
    colors_left = ['#E3F2FD', '#E8F5E9', '#FFF3E0']
    
    for (x, y, w, h, label), color in zip(left_steps, colors_left):
        box = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)
    
    # Vertical arrows left side
    ax.annotate('', xy=(2.5, 6.9), xytext=(2.5, 6.2), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(2.5, 4.9), xytext=(2.5, 4.2), arrowprops=dict(arrowstyle='->', lw=2))
    
    # RIGHT COLUMN - Steps 4-7
    right_steps = [
        (5.5, 7, 4, 1.2, 'Step 4: Get Reward\nR=1 if correct, R=0 if wrong'),
        (5.5, 5, 4, 1.2, 'Step 5: Compute Advantage\nA = R - baseline'),
        (5.5, 3, 4, 1.2, 'Step 6: Policy Gradient\ngradient = A * grad(log policy)'),
        (5.5, 1, 4, 1.2, 'Step 7: Update Weights\nweights += learning_rate * gradient'),
    ]
    
    colors_right = ['#FFEBEE', '#F3E5F5', '#E0F7FA', '#FFFDE7']
    
    for (x, y, w, h, label), color in zip(right_steps, colors_right):
        box = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)
    
    # Vertical arrows right side
    ax.annotate('', xy=(7.5, 6.9), xytext=(7.5, 6.2), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(7.5, 4.9), xytext=(7.5, 4.2), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(7.5, 2.9), xytext=(7.5, 2.2), arrowprops=dict(arrowstyle='->', lw=2))
    
    # Horizontal arrow from step 3 to step 4
    ax.annotate('', xy=(5.4, 3.6), xytext=(4.6, 3.6), arrowprops=dict(arrowstyle='->', lw=2))
    
    # KEY EQUATION BOX - separate on right
    eq_box = Rectangle((10.5, 4), 3, 4, facecolor='#FAFAFA', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(eq_box)
    ax.text(12, 7.5, 'Key Equation', fontsize=11, fontweight='bold', ha='center', color='#1976D2')
    ax.text(12, 6.5, 'Gradient of J =', fontsize=10, ha='center')
    ax.text(12, 5.8, 'E[ R * grad(log policy) ]', fontsize=10, ha='center')
    ax.text(12, 4.8, 'R = 1 (correct)', fontsize=9, ha='center', color='green')
    ax.text(12, 4.3, 'R = 0 (wrong)', fontsize=9, ha='center', color='red')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig07_reinforce.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig07 - REINFORCE (redesigned)")


def fig08_reward():
    """Reward comparison - annotation moved to not overlap with number."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rewards = ['Exact\nMatch', 'Partial\nMatch', 'Length\nPenalty', 'Combined', 'Progressive\n(slow)']
    accuracies = [29.3, 29.3, 32.4, 32.4, 43.1]
    
    colors = ['#90CAF9', '#90CAF9', '#64B5F6', '#42A5F5', '#4CAF50']
    
    bars = ax.bar(rewards, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Labels INSIDE the bars to avoid overlap
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                f'{acc:.1f}%', ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Reward Function Comparison\n(RL training, lr=2e-4)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 55)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotation ABOVE the chart, not overlapping with anything
    ax.text(4, 52, 'Best', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')
    ax.annotate('', xy=(4, 44), xytext=(4, 50),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig08_reward_functions.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig08 - Reward (fixed annotation)")


def fig10_key_findings():
    """Key findings - numbers moved to TOP-LEFT corner inside box."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(7, 7.5, 'Key Research Findings', fontsize=16, fontweight='bold', ha='center')
    
    # 6 cards in 2 rows, 3 columns
    cards = [
        # Row 1
        (0.5, 4.2, 4, 2.5, '1', 'Supervised beats RL', '74% vs 53.7%', 
         'Supervised learning with direct\nlabels outperforms REINFORCE', '#66BB6A'),
        (5, 4.2, 4, 2.5, '2', 'Optimal Learning Rate', 'lr = 2e-4',
         'Best RL accuracy achieved with\nlearning rate of 0.0002', '#42A5F5'),
        (9.5, 4.2, 4, 2.5, '3', 'Color vs Shape', '20% vs 72%',
         'RL struggles with color questions\nbut works well for shape', '#FFA726'),
        # Row 2
        (0.5, 1, 4, 2.5, '4', 'Training Helps', '0.2% to 74%',
         'Frozen baseline near random,\ntraining enables learning', '#AB47BC'),
        (5, 1, 4, 2.5, '5', 'Spatial is Hard', '24-61%',
         'Spatial reasoning most difficult\nfor all methods', '#EF5350'),
        (9.5, 1, 4, 2.5, '6', 'More Data Issue', '50K gives 61%',
         'More training data did not\nimprove over 5K baseline', '#8D6E63'),
    ]
    
    for x, y, w, h, num, title, value, desc, color in cards:
        # Card background
        box = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
        ax.add_patch(box)
        
        # Number in TOP-LEFT corner (inside the box, not on edge)
        ax.text(x + 0.3, y + h - 0.3, num, fontsize=14, fontweight='bold',
                ha='center', va='center', color=color,
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
        
        # Title - centered, near top
        ax.text(x + w/2, y + h - 0.6, title, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        
        # Value - large, centered
        ax.text(x + w/2, y + h/2, value, ha='center', va='center', 
                fontsize=14, fontweight='bold', color='white')
        
        # Description - at bottom
        ax.text(x + w/2, y + 0.5, desc, ha='center', va='center', 
                fontsize=8, color='white')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_key_findings.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig10 - Key Findings (redesigned)")


def fig02_training_pipeline():
    """Training pipeline - simpler design."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    ax.text(7, 3.6, 'Training Pipeline', fontsize=14, fontweight='bold', ha='center')
    
    # Simple horizontal boxes with good spacing
    stages = [
        (0.2, 1.2, 1.6, 1.4, 'Load\nData', '#E3F2FD'),
        (2.1, 1.2, 1.6, 1.4, 'Encode\nImage', '#BBDEFB'),
        (4.0, 1.2, 1.6, 1.4, 'Forward\nPass', '#90CAF9'),
        (5.9, 1.2, 1.6, 1.4, 'Compute\nLoss', '#64B5F6'),
        (7.8, 1.2, 1.6, 1.4, 'Backward\nPass', '#42A5F5'),
        (9.7, 1.2, 1.6, 1.4, 'Update\nWeights', '#2196F3'),
        (11.6, 1.2, 1.6, 1.4, 'Evaluate', '#1976D2'),
    ]
    
    for x, y, w, h, label, color in stages:
        box = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        # Use black text for light backgrounds, white for dark
        text_color = 'white' if color in ['#42A5F5', '#2196F3', '#1976D2'] else 'black'
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=10, fontweight='bold', color=text_color)
    
    # Arrows between boxes
    for i in range(len(stages)-1):
        x1 = stages[i][0] + stages[i][2]
        x2 = stages[i+1][0]
        y = 1.9
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Loop indicator at bottom
    ax.annotate('', xy=(0.5, 0.6), xytext=(12.5, 0.6),
                arrowprops=dict(arrowstyle='<-', color='#D32F2F', lw=2,
                               connectionstyle='arc3,rad=-0.2'))
    ax.text(6.5, 0.2, 'Repeat until convergence', fontsize=10, ha='center', color='#D32F2F')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig02_training_pipeline.png'), dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Done: fig02 - Training Pipeline (fixed)")


if __name__ == "__main__":
    print("=" * 60)
    print("COMPLETELY REDESIGNING PROBLEMATIC FIGURES")
    print("=" * 60)
    
    fig01_architecture()
    fig02_training_pipeline()
    fig05_reinforce()
    fig08_reward()
    fig10_key_findings()
    
    print("\n" + "=" * 60)
    print("ALL FIGURES REDESIGNED")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 60)
