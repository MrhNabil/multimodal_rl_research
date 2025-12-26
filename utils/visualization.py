"""
Visualization Utilities

Provides plotting functions for training curves,
accuracy comparisons, and result analysis.
"""

import os
from typing import Dict, List, Optional, Tuple
import json

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available for plotting")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def plot_training_curves(
    log_path: str,
    output_path: Optional[str] = None,
    metrics: List[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """
    Plot training curves from a log file.
    
    Args:
        log_path: Path to training log (CSV or JSONL)
        output_path: Path to save the plot
        metrics: List of metrics to plot (default: loss, accuracy, reward)
        figsize: Figure size
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting not available. Install matplotlib and seaborn.")
        return
    
    metrics = metrics or ["loss", "accuracy", "reward"]
    
    # Load data
    if log_path.endswith(".csv"):
        if not PANDAS_AVAILABLE:
            print("pandas required for CSV loading")
            return
        df = pd.read_csv(log_path)
    else:
        # JSONL
        data = []
        with open(log_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data) if PANDAS_AVAILABLE else None
    
    if df is None:
        print("Could not load data")
        return
    
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in df.columns:
            ax.plot(df["step"], df[metric], label=metric, linewidth=1.5)
            ax.set_ylabel(metric.capitalize())
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Training Step")
    plt.suptitle("Training Curves", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_accuracy_comparison(
    results: List[Dict],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot accuracy comparison across experiments.
    
    Args:
        results: List of result dictionaries with 'experiment_name' and 'accuracy'
        output_path: Path to save the plot
        figsize: Figure size
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting not available")
        return
    
    # Sort by accuracy
    results = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)
    
    names = [r.get("experiment_name", f"exp_{i}")[:20] for i, r in enumerate(results)]
    accuracies = [r.get("accuracy", 0) for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("coolwarm", len(results))
    bars = ax.barh(names, accuracies, color=colors)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}",
            va="center",
            fontsize=9,
        )
    
    ax.set_xlabel("Accuracy")
    ax.set_title("Experiment Comparison")
    ax.set_xlim(0, 1.1)
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_reward_curves(
    results_dirs: List[str],
    labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot reward curves from multiple experiments.
    
    Args:
        results_dirs: List of experiment directories
        labels: Optional labels for each experiment
        output_path: Path to save the plot
        figsize: Figure size
    """
    if not PLOTTING_AVAILABLE or not PANDAS_AVAILABLE:
        print("Plotting or pandas not available")
        return
    
    labels = labels or [os.path.basename(d) for d in results_dirs]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for result_dir, label in zip(results_dirs, labels):
        log_path = os.path.join(result_dir, "training_log.jsonl")
        if not os.path.exists(log_path):
            log_path = os.path.join(result_dir, "metrics.csv")
        
        if not os.path.exists(log_path):
            continue
        
        # Load data
        if log_path.endswith(".csv"):
            df = pd.read_csv(log_path)
        else:
            data = []
            with open(log_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        
        if "reward" in df.columns:
            ax.plot(df["step"], df["reward"], label=label, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_type_accuracy(
    per_type_results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot per-question-type accuracy comparison.
    
    Args:
        per_type_results: Dict mapping experiment name to per-type accuracy
        output_path: Path to save the plot
        figsize: Figure size
    """
    if not PLOTTING_AVAILABLE or not PANDAS_AVAILABLE:
        print("Plotting or pandas not available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(per_type_results).T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    df.plot(kind="bar", ax=ax, width=0.8)
    
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Type Accuracy Comparison")
    ax.legend(title="Question Type", loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_results_table(
    results: List[Dict],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a formatted results table.
    
    Args:
        results: List of result dictionaries
        output_path: Optional path to save as markdown
        
    Returns:
        Markdown table string
    """
    if not results:
        return "No results to display."
    
    # Define columns
    columns = [
        ("Experiment", "experiment_name"),
        ("Method", "method"),
        ("Accuracy", "accuracy"),
        ("Reward", "reward"),
        ("Seed", "seed"),
        ("LR", "learning_rate"),
    ]
    
    # Header
    header = "| " + " | ".join(c[0] for c in columns) + " |"
    separator = "|" + "|".join(["-" * (len(c[0]) + 2) for c in columns]) + "|"
    
    # Rows
    rows = []
    for r in sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True):
        row_values = []
        for col_name, col_key in columns:
            val = r.get(col_key, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            row_values.append(str(val)[:20])
        rows.append("| " + " | ".join(row_values) + " |")
    
    table = "\n".join([header, separator] + rows)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
    
    return table


def main():
    """Test visualization utilities."""
    print("Testing visualization utilities...")
    
    # Test results table
    results = [
        {"experiment_name": "frozen", "method": "frozen", "accuracy": 0.25, "seed": 42},
        {"experiment_name": "supervised", "method": "supervised", "accuracy": 0.75, "seed": 42},
        {"experiment_name": "rl", "method": "rl", "accuracy": 0.82, "seed": 42},
    ]
    
    table = generate_results_table(results)
    print("\nResults Table:")
    print(table)
    
    if PLOTTING_AVAILABLE:
        print("\nPlotting is available. Create plots with plot_* functions.")
    else:
        print("\nPlotting not available. Install matplotlib and seaborn.")


if __name__ == "__main__":
    main()
