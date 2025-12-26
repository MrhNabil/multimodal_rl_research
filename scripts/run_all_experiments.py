#!/usr/bin/env python
"""
Run All Experiments

Batch runner for all experiment configurations.
Runs experiments sequentially on CPU and collects results.
"""

import os
import argparse
import sys
import time
import subprocess
from typing import List, Dict
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_experiment_configs(config_dir: str) -> List[str]:
    """Find all experiment configuration files."""
    configs = []
    
    for filename in sorted(os.listdir(config_dir)):
        if filename.startswith("exp_") and filename.endswith(".yaml"):
            configs.append(os.path.join(config_dir, filename))
    
    return configs


def run_experiment(config_path: str, data_dir: str, output_dir: str, use_dummy: bool = False) -> Dict:
    """Run a single experiment and return results."""
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_experiment.py"),
        "--config", config_path,
        "--data_dir", data_dir,
        "--output_dir", output_dir,
    ]
    
    if use_dummy:
        cmd.append("--use_dummy")
    
    # Run experiment
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        success = True
        error_msg = None
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = e.stderr[-500:] if e.stderr else str(e)
        print(f"Experiment failed: {error_msg}")
    
    elapsed = time.time() - start_time
    
    # Load results if available
    results_path = os.path.join(output_dir, exp_name, "final_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}
    
    return {
        "experiment_name": exp_name,
        "config_path": config_path,
        "success": success,
        "error": error_msg,
        "elapsed_time": elapsed,
        **results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="config/experiments",
        help="Directory containing experiment configs",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/generated",
        help="Directory containing generated data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--use_dummy",
        action="store_true",
        help="Use dummy models for testing",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=1,
        help="Start from experiment number N",
    )
    parser.add_argument(
        "--max_experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run",
    )
    args = parser.parse_args()
    
    # Find all experiment configs
    configs = find_experiment_configs(args.config_dir)
    
    if not configs:
        print(f"No experiment configs found in {args.config_dir}")
        print("Make sure to generate configs first using the experiment config generator.")
        return
    
    # Filter experiments
    configs = configs[args.start_from - 1:]
    if args.max_experiments:
        configs = configs[:args.max_experiments]
    
    print("="*60)
    print("Batch Experiment Runner")
    print("="*60)
    print(f"Found {len(configs)} experiment(s) to run")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    all_results = []
    start_time = time.time()
    
    for i, config_path in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]", end="")
        
        results = run_experiment(
            config_path,
            args.data_dir,
            args.output_dir,
            args.use_dummy,
        )
        
        all_results.append(results)
        
        # Save intermediate results
        summary_path = os.path.join(args.output_dir, "batch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("Batch Run Complete!")
    print("="*60)
    
    successful = sum(1 for r in all_results if r.get("success", False))
    print(f"Successful: {successful}/{len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    # Print accuracy summary
    print("\nAccuracy Summary:")
    print("-"*40)
    
    sorted_results = sorted(
        [r for r in all_results if r.get("accuracy")],
        key=lambda x: x.get("accuracy", 0),
        reverse=True,
    )
    
    for r in sorted_results[:10]:
        print(f"  {r['experiment_name'][:30]}: {r.get('accuracy', 0):.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
