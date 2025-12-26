#!/usr/bin/env python
"""
Fast Batch Experiment Runner

Runs all 61 experiments with CPU-optimized settings.
Designed for quick execution (total: ~10-15 minutes for all 61 experiments).
"""

import os
import sys
import time
import json
import subprocess
from typing import List, Dict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_configs(config_dir: str) -> List[str]:
    """Find all experiment configs sorted by ID."""
    configs = []
    for f in sorted(os.listdir(config_dir)):
        if f.startswith("exp_") and f.endswith(".yaml"):
            configs.append(os.path.join(config_dir, f))
    return configs


def run_single_experiment(config_path: str, data_dir: str, output_dir: str, use_dummy: bool) -> Dict:
    """Run a single experiment and return results."""
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_experiment.py"),
        "--config", config_path,
        "--data_dir", data_dir,
        "--output_dir", output_dir,
    ]
    
    if use_dummy:
        cmd.append("--use_dummy")
    
    start = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for real models on CPU
        )
        success = result.returncode == 0
        error = None if success else result.stderr[-300:] if result.stderr else "Unknown error"
    except subprocess.TimeoutExpired:
        success = False
        error = "Timeout (30 min)"
    except Exception as e:
        success = False
        error = str(e)[:200]
    
    elapsed = time.time() - start
    
    # Try to load results from multiple possible locations
    exp_dir = os.path.join(output_dir, exp_name)
    accuracy = 0.0
    
    # Try final_results.json
    for subdir in ["training", ""]:
        results_file = os.path.join(exp_dir, subdir, "final_results.json") if subdir else os.path.join(exp_dir, "final_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    accuracy = data.get("accuracy", 0.0)
                    if accuracy > 0:
                        break
            except:
                pass
    
    # Fallback: compute accuracy from test_predictions.json
    if accuracy == 0:
        predictions_file = os.path.join(exp_dir, "test_predictions.json")
        if os.path.exists(predictions_file):
            try:
                with open(predictions_file) as f:
                    preds = json.load(f)
                    if preds:
                        correct = sum(1 for p in preds if p.get("correct", False))
                        accuracy = correct / len(preds)
            except:
                pass
    
    return {
        "experiment": exp_name,
        "success": success,
        "error": error,
        "elapsed_seconds": round(elapsed, 2),
        "accuracy": round(accuracy, 4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast Batch Experiment Runner")
    parser.add_argument("--config_dir", default="config/experiments", help="Config directory")
    parser.add_argument("--data_dir", default="data/generated", help="Data directory")
    parser.add_argument("--output_dir", default="experiments/results", help="Output directory")
    parser.add_argument("--use_dummy", action="store_true", help="Use dummy models (fast)")
    parser.add_argument("--start", type=int, default=1, help="Start from experiment N")
    parser.add_argument("--end", type=int, default=None, help="End at experiment N")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    args = parser.parse_args()
    
    configs = find_configs(args.config_dir)
    
    if args.end:
        configs = configs[args.start-1:args.end]
    else:
        configs = configs[args.start-1:]
    
    print("=" * 70)
    print("FAST BATCH EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Config dir: {args.config_dir}")
    print(f"Experiments: {len(configs)}")
    print(f"Mode: {'Dummy models (FAST)' if args.use_dummy else 'Real models'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    start_time = time.time()
    
    for i, config in enumerate(configs, 1):
        exp_name = os.path.splitext(os.path.basename(config))[0]
        print(f"\n[{i:02d}/{len(configs)}] {exp_name}...", end=" ", flush=True)
        
        result = run_single_experiment(
            config, args.data_dir, args.output_dir, args.use_dummy
        )
        all_results.append(result)
        
        if result["success"]:
            print(f"OK ({result['elapsed_seconds']:.1f}s, acc={result['accuracy']:.3f})")
        else:
            print(f"FAIL ({result['error'][:50] if result['error'] else 'Unknown'})")
        
        # Save intermediate results
        summary_path = os.path.join(args.output_dir, "batch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r["success"])
    
    print("\n" + "=" * 70)
    print("BATCH RUN COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}/{len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results: {args.output_dir}/batch_summary.json")
    print("=" * 70)
    
    # Quick accuracy summary
    if any(r.get("accuracy", 0) > 0 for r in all_results):
        print("\nTop 5 by Accuracy:")
        sorted_results = sorted(all_results, key=lambda x: x.get("accuracy", 0), reverse=True)
        for r in sorted_results[:5]:
            print(f"   {r['experiment'][:40]}: {r.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    main()
