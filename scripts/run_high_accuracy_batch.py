#!/usr/bin/env python
"""Run all experiments with the HIGH-ACCURACY model."""

import os
import sys
import time
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from data.dataset import MultimodalDataset, create_dataloader
from models.high_accuracy_vqa import create_high_accuracy_model
from utils.helpers import set_seed, load_config, merge_configs


def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    per_type = {}
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch.images.to(device)
            output = model(images, batch.questions, mode="greedy")
            
            for pred, gt, qtype in zip(output.texts, batch.answers, batch.question_types):
                if qtype not in per_type:
                    per_type[qtype] = [0, 0]
                per_type[qtype][1] += 1
                
                if pred.lower() == gt.lower().strip():
                    correct += 1
                    per_type[qtype][0] += 1
                total += 1
    
    return correct / total if total > 0 else 0, per_type


def run_experiment(config_path, data_dir, output_dir, device):
    """Run single experiment with high-accuracy model."""
    
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "base_config.yaml")
    config = merge_configs(load_config(base_path), load_config(config_path))
    
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    set_seed(config.get("seed", 42))
    
    # Load data
    train_ds = MultimodalDataset(data_dir, "train")
    val_ds = MultimodalDataset(data_dir, "val")
    test_ds = MultimodalDataset(data_dir, "test")
    
    batch_size = config.get("training", {}).get("batch_size", 64)
    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Create HIGH-ACCURACY model
    model = create_high_accuracy_model(device)
    lr = config.get("training", {}).get("optimizer", {}).get("learning_rate", 5e-4)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=lr)
    
    max_steps = config.get("training", {}).get("max_steps", 500)
    method = config.get("training", {}).get("method", "supervised")
    
    model.train()
    step = 0
    best_val = 0.0
    
    pbar = tqdm(total=max_steps, desc=f"{exp_name}")
    start_time = time.time()
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            images = batch.images.to(device)
            
            if method == "supervised":
                loss = model.compute_supervised_loss(images, batch.questions, batch.answers)
            else:  # RL
                output = model(images, batch.questions, mode="sample")
                rewards = torch.tensor([
                    1.0 if p.lower() == a.lower().strip() else 0.0
                    for p, a in zip(output.texts, batch.answers)
                ], device=device)
                advantages = rewards - rewards.mean()
                loss = -(output.log_probs * advantages).mean() - 0.01 * output.entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
            optimizer.step()
            
            step += 1
            pbar.update(1)
            
            if step % 100 == 0:
                val_acc, _ = evaluate(model, val_loader, device)
                best_val = max(best_val, val_acc)
                pbar.set_postfix({"val": f"{val_acc:.2%}"})
                model.train()
    
    pbar.close()
    
    test_acc, per_type = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time
    
    # Save results
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    results = {
        "experiment_name": exp_name,
        "method": method,
        "accuracy": test_acc,
        "best_val_accuracy": best_val,
        "per_type": {k: v[0]/v[1] if v[1] > 0 else 0 for k, v in per_type.items()},
        "elapsed_seconds": elapsed,
        "model": "HighAccuracyVQA",
    }
    
    with open(os.path.join(exp_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ {exp_name}: {test_acc:.2%} ({elapsed:.1f}s)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", default="config/experiments")
    parser.add_argument("--data_dir", default="data/generated")
    parser.add_argument("--output_dir", default="experiments/results_high_acc")
    parser.add_argument("--start_from", type=int, default=1)
    parser.add_argument("--max_experiments", type=int, default=None)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    configs = sorted([
        os.path.join(args.config_dir, f)
        for f in os.listdir(args.config_dir)
        if f.startswith("exp_") and f.endswith(".yaml")
    ])[args.start_from - 1:]
    
    if args.max_experiments:
        configs = configs[:args.max_experiments]
    
    print("=" * 60)
    print("HIGH-ACCURACY EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Running {len(configs)} experiments from #{args.start_from}")
    print(f"Device: {device}")
    print("=" * 60)
    
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        try:
            r = run_experiment(cfg, args.data_dir, args.output_dir, device)
            r["success"] = True
            results.append(r)
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({"config": cfg, "success": False, "error": str(e)})
        
        with open(os.path.join(args.output_dir, "batch_summary.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    successful = [r for r in results if r.get("success")]
    print(f"Complete: {len(successful)}/{len(results)}")
    if successful:
        best = max(successful, key=lambda x: x.get("accuracy", 0))
        print(f"Best: {best['experiment_name']} = {best['accuracy']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
