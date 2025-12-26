#!/usr/bin/env python
"""
Analyze Results

Aggregates results from multiple experiments and generates
summary tables, plots, and comparisons.
"""

import os
import argparse
import sys
import json
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging import AggregateLogger
from utils.visualization import (
    plot_accuracy_comparison,
    plot_reward_curves,
    plot_per_type_accuracy,
    generate_results_table,
    PLOTTING_AVAILABLE,
)


def collect_all_results(results_dir: str) -> List[Dict]:
    """Collect results from all experiments."""
    results = []
    
    for exp_name in sorted(os.listdir(results_dir)):
        exp_dir = os.path.join(results_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        
        # Load final results
        final_path = os.path.join(exp_dir, "final_results.json")
        if os.path.exists(final_path):
            with open(final_path, "r") as f:
                result = json.load(f)
                result["experiment_name"] = exp_name
                results.append(result)
        
        # Load evaluation results
        eval_path = os.path.join(exp_dir, "test_metrics.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                eval_result = json.load(f)
                for r in results:
                    if r["experiment_name"] == exp_name:
                        r.update(eval_result)
    
    return results


def analyze_by_method(results: List[Dict]) -> Dict[str, Dict]:
    """Analyze results grouped by training method."""
    methods = {}
    
    for r in results:
        method = r.get("method", "unknown")
        if method not in methods:
            methods[method] = {
                "accuracies": [],
                "count": 0,
            }
        
        methods[method]["accuracies"].append(r.get("accuracy", 0))
        methods[method]["count"] += 1
    
    # Compute statistics
    for method, data in methods.items():
        accs = data["accuracies"]
        data["mean_accuracy"] = sum(accs) / len(accs) if accs else 0
        data["min_accuracy"] = min(accs) if accs else 0
        data["max_accuracy"] = max(accs) if accs else 0
    
    return methods


def analyze_by_hyperparameter(results: List[Dict], param_name: str) -> Dict:
    """Analyze results grouped by a specific hyperparameter."""
    groups = {}
    
    for r in results:
        param_value = r.get(param_name, "unknown")
        if param_value not in groups:
            groups[param_value] = []
        groups[param_value].append(r.get("accuracy", 0))
    
    # Compute statistics
    analysis = {}
    for value, accs in groups.items():
        analysis[str(value)] = {
            "mean": sum(accs) / len(accs) if accs else 0,
            "count": len(accs),
            "min": min(accs) if accs else 0,
            "max": max(accs) if accs else 0,
        }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/analysis",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires matplotlib)",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Experiment Analysis")
    print("="*60)
    
    # Collect results
    results = collect_all_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Found {len(results)} experiment results\n")
    
    # Analysis by method
    print("Analysis by Training Method:")
    print("-"*40)
    method_analysis = analyze_by_method(results)
    for method, data in sorted(method_analysis.items()):
        print(f"  {method}:")
        print(f"    Count: {data['count']}")
        print(f"    Mean accuracy: {data['mean_accuracy']:.4f}")
        print(f"    Min/Max: {data['min_accuracy']:.4f} / {data['max_accuracy']:.4f}")
    
    # Analysis by learning rate
    print("\nAnalysis by Learning Rate:")
    print("-"*40)
    lr_analysis = analyze_by_hyperparameter(results, "learning_rate")
    for lr, data in sorted(lr_analysis.items(), key=lambda x: float(x[0]) if x[0] != "unknown" else 0):
        print(f"  {lr}: mean={data['mean']:.4f} (n={data['count']})")
    
    # Analysis by reward type
    print("\nAnalysis by Reward Type:")
    print("-"*40)
    reward_analysis = analyze_by_hyperparameter(results, "reward_type")
    for reward, data in sorted(reward_analysis.items()):
        print(f"  {reward}: mean={data['mean']:.4f} (n={data['count']})")
    
    # Generate summary table
    print("\nTop 10 Experiments by Accuracy:")
    print("-"*40)
    sorted_results = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"  {i}. {r['experiment_name'][:30]}: {r.get('accuracy', 0):.4f}")
    
    # Save detailed analysis
    analysis_output = {
        "num_experiments": len(results),
        "by_method": method_analysis,
        "by_learning_rate": lr_analysis,
        "by_reward_type": reward_analysis,
        "top_10": sorted_results[:10],
    }
    
    analysis_path = os.path.join(args.output_dir, "analysis_summary.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis_output, f, indent=2, default=str)
    
    # Generate markdown table
    table = generate_results_table(results)
    table_path = os.path.join(args.output_dir, "results_table.md")
    with open(table_path, "w") as f:
        f.write("# Experiment Results\n\n")
        f.write(table)
    
    # Generate plots
    if args.plot and PLOTTING_AVAILABLE:
        print("\nGenerating plots...")
        
        # Accuracy comparison
        plot_accuracy_comparison(
            sorted_results[:20],
            output_path=os.path.join(args.output_dir, "accuracy_comparison.png"),
        )
        
        # Per-type accuracy (if available)
        per_type_results = {}
        for r in sorted_results[:5]:
            if "per_type_accuracy" in r:
                per_type_results[r["experiment_name"]] = r["per_type_accuracy"]
        
        if per_type_results:
            plot_per_type_accuracy(
                per_type_results,
                output_path=os.path.join(args.output_dir, "per_type_accuracy.png"),
            )
    
    print(f"\nAnalysis saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
