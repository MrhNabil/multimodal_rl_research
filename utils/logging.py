"""
Experiment Logging Utilities

Provides structured logging to CSV and JSON formats
for tracking experiments and reproducibility.
"""

import os
import json
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration snapshot for an experiment."""
    experiment_name: str
    method: str  # "frozen", "supervised", "rl"
    seed: int
    learning_rate: float
    batch_size: int
    reward_type: Optional[str] = None
    baseline_type: Optional[str] = None
    temperature: Optional[float] = None
    entropy_coef: Optional[float] = None
    extra: Optional[Dict] = None


class ExperimentLogger:
    """
    Logger for tracking experiments.
    
    Saves configurations, metrics, and results in
    both CSV (for easy analysis) and JSON (for completeness).
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            config: Optional configuration dictionary
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.config = config or {}
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.run_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize log files
        self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
        self.json_log_path = os.path.join(self.run_dir, "log.jsonl")
        self.config_path = os.path.join(self.run_dir, "config.json")
        
        # Save config
        self._save_config()
        
        # Initialize CSV
        self._csv_initialized = False
    
    def _save_config(self):
        """Save experiment configuration."""
        config_with_meta = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            **self.config,
        }
        with open(self.config_path, "w") as f:
            json.dump(config_with_meta, f, indent=2)
    
    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
    ):
        """
        Log metrics for a training step.
        
        Args:
            step: Training step number
            metrics: Dictionary of metrics to log
        """
        # Add step and timestamp
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        
        # Log to JSON
        with open(self.json_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Log to CSV
        self._log_csv(log_entry)
    
    def _log_csv(self, entry: Dict):
        """Log entry to CSV file."""
        if not self._csv_initialized:
            # Write header
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                writer.writeheader()
            self._csv_initialized = True
        
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            writer.writerow(entry)
    
    def log_evaluation(
        self,
        split: str,
        metrics: Dict[str, Any],
    ):
        """
        Log evaluation results.
        
        Args:
            split: Dataset split name
            metrics: Dictionary of evaluation metrics
        """
        eval_path = os.path.join(self.run_dir, f"eval_{split}.json")
        with open(eval_path, "w") as f:
            json.dump({
                "split": split,
                "timestamp": datetime.now().isoformat(),
                **metrics,
            }, f, indent=2)
    
    def log_final_results(
        self,
        results: Dict[str, Any],
    ):
        """
        Log final experiment results.
        
        Args:
            results: Dictionary of final results
        """
        results_path = os.path.join(self.run_dir, "final_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "timestamp": datetime.now().isoformat(),
                **results,
            }, f, indent=2)
    
    def get_run_dir(self) -> str:
        """Get the run directory path."""
        return self.run_dir


class AggregateLogger:
    """
    Aggregates results from multiple experiments.
    
    Used for generating summary tables and comparisons.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize aggregate logger.
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
    
    def collect_results(self) -> List[Dict]:
        """
        Collect results from all experiments.
        
        Returns:
            List of result dictionaries
        """
        results = []
        
        for exp_name in os.listdir(self.results_dir):
            exp_dir = os.path.join(self.results_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue
            
            # Load final results
            results_path = os.path.join(exp_dir, "final_results.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    result = json.load(f)
                    result["experiment_name"] = exp_name
                    results.append(result)
        
        return results
    
    def generate_summary_table(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a summary table of all experiments.
        
        Args:
            output_path: Optional path to save the table
            
        Returns:
            Table as a string
        """
        results = self.collect_results()
        
        if not results:
            return "No results found."
        
        # Generate table header
        columns = ["experiment", "method", "accuracy", "seed", "lr"]
        header = " | ".join(columns)
        separator = "-" * len(header)
        
        # Generate rows
        rows = []
        for r in sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True):
            row = " | ".join([
                str(r.get("experiment_name", ""))[:20],
                str(r.get("method", ""))[:15],
                f"{r.get('accuracy', 0):.4f}",
                str(r.get("seed", "")),
                str(r.get("learning_rate", "")),
            ])
            rows.append(row)
        
        table = "\n".join([header, separator] + rows)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(table)
        
        return table


def main():
    """Test logging utilities."""
    print("Testing ExperimentLogger...")
    
    logger = ExperimentLogger(
        log_dir="experiments/results",
        experiment_name="test_experiment",
        config={"learning_rate": 1e-4, "batch_size": 32},
    )
    
    # Log some steps
    for step in range(5):
        logger.log_step(step, {
            "loss": 1.0 / (step + 1),
            "accuracy": 0.5 + step * 0.1,
        })
    
    # Log evaluation
    logger.log_evaluation("val", {"accuracy": 0.85})
    
    # Log final results
    logger.log_final_results({
        "accuracy": 0.85,
        "method": "rl",
    })
    
    print(f"Logs saved to: {logger.get_run_dir()}")


if __name__ == "__main__":
    main()
