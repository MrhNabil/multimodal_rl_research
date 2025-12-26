#!/usr/bin/env python
"""
Experimental Grid Runner - Shows How Hyperparameters Affect Accuracy

This script runs a controlled grid of experiments to demonstrate:
1. Effect of TRAINING STEPS (100, 300, 500, 1000, 2000)
2. Effect of LEARNING RATE (1e-4, 5e-4, 1e-3, 5e-3)
3. Effect of METHOD (frozen, supervised, RL)

Results are saved to a CSV for easy plotting and presentation.
"""

import os
import sys
import time
import json
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.helpers import set_seed
from data.dataset import MultimodalDataset, create_dataloader
from models.vision import create_vision_encoder
from models.projection import create_projection_layer
from models.fast_reasoning import FastVQAReasoner, FastMultimodalVQA


def create_model(device="cuda"):
    """Create model on GPU."""
    vision_encoder = create_vision_encoder("ViT-B-32", "openai", device=device, use_dummy=False)
    vision_dim = vision_encoder.get_embedding_dim()
    
    reasoner = FastVQAReasoner(input_dim=vision_dim, hidden_dim=256, num_layers=2, device=device)
    projection_layer = create_projection_layer(input_dim=vision_dim, output_dim=vision_dim, use_hidden=False).to(device)
    
    model = FastMultimodalVQA(vision_encoder, reasoner, projection_layer).to(device)
    return model


def evaluate(model, dataloader, device):
    """Quick evaluation."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch.images.to(device)
            preds = model(images, batch.questions, mode="greedy").texts
            for p, a in zip(preds, batch.answers):
                if p.lower() == a.lower().strip():
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0


def train_and_evaluate(
    method: str,
    learning_rate: float,
    max_steps: int,
    device: str = "cuda",
    seed: int = 42,
):
    """Train with specific hyperparameters and return accuracy."""
    set_seed(seed)
    
    # Load data
    train_ds = MultimodalDataset("data/generated", "train")
    val_ds = MultimodalDataset("data/generated", "val")
    test_ds = MultimodalDataset("data/generated", "test")
    
    train_loader = create_dataloader(train_ds, batch_size=64, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=64, shuffle=False)
    
    # Create model
    model = create_model(device)
    
    # Frozen baseline - just evaluate, no training
    if method == "frozen":
        test_acc = evaluate(model, test_loader, device)
        return {
            "method": method,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "test_accuracy": test_acc,
            "val_accuracy": evaluate(model, val_loader, device),
        }
    
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=learning_rate)
    
    model.train()
    step = 0
    best_val = 0.0
    
    pbar = tqdm(total=max_steps, desc=f"{method} lr={learning_rate} steps={max_steps}")
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            images = batch.images.to(device)
            questions = batch.questions
            answers = batch.answers
            
            if method == "supervised":
                # Cross-entropy loss
                loss = model.compute_supervised_loss(images, questions, answers)
            else:  # RL
                # REINFORCE
                output = model(images, questions, mode="sample", temperature=1.0)
                rewards = torch.tensor([
                    1.0 if p.lower() == a.lower().strip() else 0.0
                    for p, a in zip(output.texts, answers)
                ], device=device)
                
                baseline = rewards.mean()
                advantages = rewards - baseline
                pg_loss = -(output.log_probs.squeeze(-1) * advantages).mean()
                entropy_loss = -0.01 * output.entropy.mean()
                loss = pg_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
            optimizer.step()
            
            step += 1
            pbar.update(1)
            
            # Periodic evaluation
            if step % 200 == 0 or step == max_steps:
                val_acc = evaluate(model, val_loader, device)
                best_val = max(best_val, val_acc)
                pbar.set_postfix({"val_acc": f"{val_acc:.3f}"})
                model.train()
    
    pbar.close()
    
    # Final test evaluation
    test_acc = evaluate(model, test_loader, device)
    
    return {
        "method": method,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "test_accuracy": test_acc,
        "val_accuracy": best_val,
    }


def run_experimental_grid():
    """Run full experimental grid."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("EXPERIMENTAL GRID: Hyperparameter Effect on Accuracy")
    print("=" * 70)
    print(f"Device: {device}")
    print("=" * 70)
    
    # Define experimental grid
    experiments = []
    
    # Experiment 1: Effect of Training Steps (fixed LR=1e-3)
    # Range: 700-3000 steps
    print("\n[1/4] Effect of TRAINING STEPS (LR=1e-3)")
    for steps in [700, 1000, 1500, 2000, 2500, 3000]:
        for method in ["supervised", "rl"]:
            experiments.append({
                "method": method,
                "learning_rate": 1e-3,
                "max_steps": steps,
                "category": "steps_effect"
            })
    
    # Experiment 2: Effect of Learning Rate (fixed steps=1000) - 10 experiments
    print("[2/4] Effect of LEARNING RATE (steps=1000) - 10 experiments")
    for lr in [1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2]:
        experiments.append({
            "method": "supervised",
            "learning_rate": lr,
            "max_steps": 1000,
            "category": "lr_effect_1000"
        })
    
    # Experiment 3: RL with different learning rates at 1000 steps
    print("[3/4] RL Learning Rate sweep (steps=1000)")
    for lr in [1e-4, 5e-4, 1e-3, 3e-3, 5e-3]:
        experiments.append({
            "method": "rl",
            "learning_rate": lr,
            "max_steps": 1000,
            "category": "lr_effect_rl"
        })
    
    # Experiment 4: Best configs with more steps (2000-3000)
    print("[4/4] High-step experiments (2000-3000)")
    for steps in [2000, 2500, 3000]:
        for lr in [5e-4, 1e-3, 2e-3]:
            experiments.append({
                "method": "supervised",
                "learning_rate": lr,
                "max_steps": steps,
                "category": "high_steps"
            })
    
    # Frozen baseline
    experiments.append({
        "method": "frozen",
        "learning_rate": 0,
        "max_steps": 0,
        "category": "baseline"
    })
    
    print(f"\nTotal experiments: {len(experiments)}")
    print("=" * 70)
    
    # Output directory
    output_dir = "experiments/grid_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiments
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp['category']}: {exp['method']} lr={exp['learning_rate']} steps={exp['max_steps']}")
        
        start_time = time.time()
        
        try:
            result = train_and_evaluate(
                method=exp["method"],
                learning_rate=exp["learning_rate"],
                max_steps=exp["max_steps"],
                device=device,
            )
            result["category"] = exp["category"]
            result["elapsed_seconds"] = time.time() - start_time
            result["success"] = True
            
            print(f"  ✓ Test Accuracy: {result['test_accuracy']:.4f} ({result['elapsed_seconds']:.1f}s)")
            
        except Exception as e:
            result = {
                "method": exp["method"],
                "learning_rate": exp["learning_rate"],
                "max_steps": exp["max_steps"],
                "category": exp["category"],
                "test_accuracy": 0,
                "success": False,
                "error": str(e),
            }
            print(f"  ✗ Failed: {e}")
        
        results.append(result)
        
        # Save intermediate results
        with open(os.path.join(output_dir, "grid_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    # Save to CSV for easy analysis
    csv_path = os.path.join(output_dir, "grid_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "method", "learning_rate", "max_steps", 
            "test_accuracy", "val_accuracy", "elapsed_seconds", "success"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "category": r.get("category", ""),
                "method": r.get("method", ""),
                "learning_rate": r.get("learning_rate", 0),
                "max_steps": r.get("max_steps", 0),
                "test_accuracy": r.get("test_accuracy", 0),
                "val_accuracy": r.get("val_accuracy", 0),
                "elapsed_seconds": r.get("elapsed_seconds", 0),
                "success": r.get("success", False),
            })
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n--- Effect of Training Steps (LR=1e-3) ---")
    print(f"{'Steps':>6} | {'Supervised':>12} | {'RL':>12}")
    print("-" * 35)
    for steps in [700, 1000, 1500, 2000, 2500, 3000]:
        sup = [r for r in results if r.get("max_steps") == steps and r.get("method") == "supervised" and r.get("learning_rate") == 1e-3]
        rl = [r for r in results if r.get("max_steps") == steps and r.get("method") == "rl" and r.get("learning_rate") == 1e-3]
        sup_acc = sup[0]["test_accuracy"] if sup else 0
        rl_acc = rl[0]["test_accuracy"] if rl else 0
        print(f"{steps:>6} | {sup_acc:>11.2%} | {rl_acc:>11.2%}")
    
    print("\n--- Effect of Learning Rate (1000 steps, Supervised) ---")
    print(f"{'LR':>10} | {'Accuracy':>12}")
    print("-" * 26)
    for lr in [1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2]:
        sup = [r for r in results if r.get("learning_rate") == lr and r.get("method") == "supervised" and r.get("max_steps") == 1000]
        sup_acc = sup[0]["test_accuracy"] if sup else 0
        print(f"{lr:>10.0e} | {sup_acc:>11.2%}")
    
    print("\n--- Effect of Learning Rate (1000 steps, RL) ---")
    print(f"{'LR':>10} | {'Accuracy':>12}")
    print("-" * 26)
    for lr in [1e-4, 5e-4, 1e-3, 3e-3, 5e-3]:
        rl = [r for r in results if r.get("learning_rate") == lr and r.get("method") == "rl" and r.get("max_steps") == 1000]
        rl_acc = rl[0]["test_accuracy"] if rl else 0
        print(f"{lr:>10.0e} | {rl_acc:>11.2%}")
    
    print("\n--- High-Step Experiments (Supervised) ---")
    print(f"{'Steps':>6} | {'LR 5e-4':>12} | {'LR 1e-3':>12} | {'LR 2e-3':>12}")
    print("-" * 50)
    for steps in [2000, 2500, 3000]:
        acc_1 = [r for r in results if r.get("max_steps") == steps and r.get("learning_rate") == 5e-4]
        acc_2 = [r for r in results if r.get("max_steps") == steps and r.get("learning_rate") == 1e-3 and r.get("category") == "high_steps"]
        acc_3 = [r for r in results if r.get("max_steps") == steps and r.get("learning_rate") == 2e-3]
        a1 = acc_1[0]["test_accuracy"] if acc_1 else 0
        a2 = acc_2[0]["test_accuracy"] if acc_2 else 0
        a3 = acc_3[0]["test_accuracy"] if acc_3 else 0
        print(f"{steps:>6} | {a1:>11.2%} | {a2:>11.2%} | {a3:>11.2%}")
    
    frozen = [r for r in results if r.get("method") == "frozen"]
    if frozen:
        print(f"\n--- Frozen Baseline (no training) ---")
        print(f"Accuracy: {frozen[0]['test_accuracy']:.2%}")
    
    # Find best result
    valid_results = [r for r in results if r.get("success") and r.get("method") != "frozen"]
    if valid_results:
        best = max(valid_results, key=lambda x: x.get("test_accuracy", 0))
        print(f"\n--- BEST RESULT ---")
        print(f"Method: {best['method']}, LR: {best['learning_rate']}, Steps: {best['max_steps']}")
        print(f"Accuracy: {best['test_accuracy']:.2%}")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}/")
    print("  - grid_results.json (full data)")
    print("  - grid_results.csv (for Excel/plotting)")
    print("=" * 70)


if __name__ == "__main__":
    run_experimental_grid()
