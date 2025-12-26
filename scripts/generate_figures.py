#!/usr/bin/env python
"""
Generate Publication-Quality Graphs from Experiment Results

Creates figures for:
1. Method Comparison (Frozen vs Supervised vs RL)
2. Learning Rate Effect
3. Reward Function Comparison
4. Question Type Analysis
5. Training Curves
"""

import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})

# Colors for consistent styling
COLORS = {
    'frozen': '#95a5a6',    # Gray
    'supervised': '#3498db', # Blue
    'rl': '#e74c3c',         # Red
    'best': '#27ae60',       # Green
}


def load_results(results_dir):
    """Load all experiment results from directory."""
    results = []
    
    for exp_name in sorted(os.listdir(results_dir)):
        exp_dir = os.path.join(results_dir, exp_name)
        results_file = os.path.join(exp_dir, "final_results.json")
        
        if os.path.exists(results_file):
            with open(results_file) as f:
                data = json.load(f)
                data['experiment_name'] = exp_name
                results.append(data)
    
    # Also check for batch_summary.json
    batch_file = os.path.join(results_dir, "batch_summary.json")
    if os.path.exists(batch_file) and not results:
        with open(batch_file) as f:
            results = json.load(f)
    
    return results


def plot_method_comparison(results, output_dir):
    """Plot comparison of Frozen vs Supervised vs RL."""
    methods = {'frozen': [], 'supervised': [], 'rl': []}
    
    for r in results:
        method = r.get('method', 'rl')
        acc = r.get('accuracy', 0)
        if method in methods:
            methods[method].append(acc)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = ['Frozen\n(No Training)', 'Supervised\nLearning', 'Reinforcement\nLearning']
    y = [
        np.mean(methods['frozen']) if methods['frozen'] else 0,
        np.mean(methods['supervised']) if methods['supervised'] else 0,
        np.mean(methods['rl']) if methods['rl'] else 0,
    ]
    yerr = [
        np.std(methods['frozen']) if len(methods['frozen']) > 1 else 0,
        np.std(methods['supervised']) if len(methods['supervised']) > 1 else 0,
        np.std(methods['rl']) if len(methods['rl']) > 1 else 0,
    ]
    
    colors = [COLORS['frozen'], COLORS['supervised'], COLORS['rl']]
    bars = ax.bar(x, [v * 100 for v in y], yerr=[e * 100 for e in yerr], 
                  color=colors, capsize=5, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, y):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Training Method Comparison\nFrozen vs Supervised vs RL', fontweight='bold')
    ax.set_ylim(0, max(y) * 100 * 1.2 + 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: fig1_method_comparison.png")


def plot_learning_rate_effect(results, output_dir):
    """Plot accuracy vs learning rate."""
    lr_results = defaultdict(list)
    
    for r in results:
        name = r.get('experiment_name', '')
        if 'lr_' in name:
            lr = r.get('learning_rate', 0)
            acc = r.get('accuracy', 0)
            if lr > 0:
                lr_results[lr].append(acc)
    
    if not lr_results:
        print("⚠ No learning rate experiments found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lrs = sorted(lr_results.keys())
    means = [np.mean(lr_results[lr]) for lr in lrs]
    stds = [np.std(lr_results[lr]) if len(lr_results[lr]) > 1 else 0 for lr in lrs]
    
    ax.errorbar(range(len(lrs)), [m * 100 for m in means], 
                yerr=[s * 100 for s in stds], 
                marker='o', markersize=10, linewidth=2, capsize=5,
                color=COLORS['rl'], markerfacecolor='white', markeredgewidth=2)
    
    # Highlight best
    best_idx = np.argmax(means)
    ax.scatter([best_idx], [means[best_idx] * 100], 
               s=200, color=COLORS['best'], zorder=5, marker='*')
    
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Effect of Learning Rate on Model Performance', fontweight='bold')
    
    # Add annotation for best
    ax.annotate(f'Best: {lrs[best_idx]:.0e}\n({means[best_idx]*100:.1f}%)',
                xy=(best_idx, means[best_idx] * 100),
                xytext=(best_idx + 0.5, means[best_idx] * 100 + 5),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: fig2_learning_rate.png")


def plot_reward_comparison(results, output_dir):
    """Plot comparison of different reward functions."""
    reward_results = {}
    
    for r in results:
        name = r.get('experiment_name', '')
        if 'reward_' in name:
            # Extract reward type from name
            reward_type = name.split('reward_')[1].split('_')[0] if 'reward_' in name else 'unknown'
            acc = r.get('accuracy', 0)
            if reward_type not in reward_results:
                reward_results[reward_type] = []
            reward_results[reward_type].append(acc)
    
    if not reward_results:
        print("⚠ No reward function experiments found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rewards = list(reward_results.keys())
    means = [np.mean(reward_results[r]) * 100 for r in rewards]
    stds = [np.std(reward_results[r]) * 100 if len(reward_results[r]) > 1 else 0 for r in rewards]
    
    bars = ax.barh(rewards, means, xerr=stds, capsize=4,
                   color=COLORS['rl'], edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_ylabel('Reward Function')
    ax.set_title('Effect of Reward Function on RL Performance', fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, means):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_reward_functions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: fig3_reward_functions.png")


def plot_question_type_analysis(results, output_dir):
    """Plot accuracy by question type."""
    # Collect per-type accuracies
    type_accs = defaultdict(list)
    
    for r in results:
        per_type = r.get('per_type_accuracy', {})
        for qtype, acc in per_type.items():
            type_accs[qtype].append(acc)
    
    if not type_accs:
        print("⚠ No per-type accuracy data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    qtypes = ['color', 'shape', 'count', 'spatial']
    qtypes = [q for q in qtypes if q in type_accs]  # Only include present types
    
    x = np.arange(len(qtypes))
    width = 0.6
    
    means = [np.mean(type_accs[q]) * 100 for q in qtypes]
    stds = [np.std(type_accs[q]) * 100 if len(type_accs[q]) > 1 else 0 for q in qtypes]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(qtypes)))
    bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlabel('Question Type')
    ax.set_xticks(x)
    ax.set_xticklabels([q.capitalize() for q in qtypes])
    ax.set_title('Model Performance by Question Type', fontweight='bold')
    ax.set_ylim(0, max(means) * 1.2 + 10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_question_types.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: fig4_question_types.png")


def plot_experiment_summary(results, output_dir):
    """Plot all experiments sorted by accuracy."""
    if not results:
        print("⚠ No results to plot")
        return
    
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True)[:20]  # Top 20
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    names = [r.get('experiment_name', 'unknown')[-25:] for r in sorted_results]  # Truncate names
    accs = [r.get('accuracy', 0) * 100 for r in sorted_results]
    methods = [r.get('method', 'rl') for r in sorted_results]
    
    colors = [COLORS.get(m, COLORS['rl']) for m in methods]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, accs, color=colors, edgecolor='black', linewidth=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Top 20 Experiments by Accuracy', fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, accs):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['frozen'], label='Frozen'),
        Patch(facecolor=COLORS['supervised'], label='Supervised'),
        Patch(facecolor=COLORS['rl'], label='RL'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_experiment_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: fig5_experiment_summary.png")


def generate_summary_table(results, output_dir):
    """Generate a summary table as text file."""
    if not results:
        return
    
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True)
    
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Best accuracy: {sorted_results[0].get('accuracy', 0)*100:.2f}%\n")
        f.write(f"Best experiment: {sorted_results[0].get('experiment_name', 'unknown')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{'Experiment':<40} | {'Method':>12} | {'Accuracy':>10}\n")
        f.write("-" * 80 + "\n")
        
        for r in sorted_results[:30]:
            name = r.get('experiment_name', 'unknown')[:40]
            method = r.get('method', 'rl')
            acc = r.get('accuracy', 0) * 100
            f.write(f"{name:<40} | {method:>12} | {acc:>9.2f}%\n")
        
        f.write("-" * 80 + "\n")
    
    print("✓ Created: results_summary.txt")


def main():
    parser = argparse.ArgumentParser(description='Generate graphs from experiment results')
    parser.add_argument('--results_dir', default='experiments/results_fixed',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', default='experiments/figures',
                       help='Directory to save figures')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    print(f"Results from: {args.results_dir}")
    print(f"Figures saved to: {args.output_dir}")
    print("=" * 60)
    
    # Load results
    results = load_results(args.results_dir)
    print(f"\nLoaded {len(results)} experiment results")
    
    if not results:
        print("❌ No results found! Make sure experiments have completed.")
        return
    
    # Generate all figures
    print("\nGenerating figures...")
    plot_method_comparison(results, args.output_dir)
    plot_learning_rate_effect(results, args.output_dir)
    plot_reward_comparison(results, args.output_dir)
    plot_question_type_analysis(results, args.output_dir)
    plot_experiment_summary(results, args.output_dir)
    generate_summary_table(results, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ All figures generated!")
    print(f"Check: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
