#!/usr/bin/env python
"""Train and test the high-accuracy VQA model."""

import os
import sys
import time
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from data.dataset import MultimodalDataset, create_dataloader
from models.high_accuracy_vqa import create_high_accuracy_model
from utils.helpers import set_seed


def evaluate(model, dataloader, device):
    """Evaluate accuracy."""
    model.eval()
    correct = total = 0
    per_type = {"color": [0, 0], "shape": [0, 0], "count": [0, 0], "spatial": [0, 0]}
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch.images.to(device)
            output = model(images, batch.questions, mode="greedy")
            
            for pred, gt, qtype in zip(output.texts, batch.answers, batch.question_types):
                gt_clean = gt.lower().strip()
                pred_clean = pred.lower().strip()
                
                if qtype in per_type:
                    per_type[qtype][1] += 1
                    if pred_clean == gt_clean:
                        per_type[qtype][0] += 1
                
                if pred_clean == gt_clean:
                    correct += 1
                total += 1
    
    overall = correct / total if total > 0 else 0
    by_type = {t: (c[0]/c[1] if c[1] > 0 else 0) for t, c in per_type.items()}
    
    return overall, by_type


def train(max_steps=1000, lr=1e-3, method="supervised"):
    """Train the model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    print("=" * 60)
    print(f"HIGH-ACCURACY VQA TRAINING")
    print(f"Method: {method}, LR: {lr}, Steps: {max_steps}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load data
    train_ds = MultimodalDataset("data/generated", "train")
    val_ds = MultimodalDataset("data/generated", "val")
    test_ds = MultimodalDataset("data/generated", "test")
    
    train_loader = create_dataloader(train_ds, batch_size=64, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=64, shuffle=False)
    
    # Create model
    model = create_high_accuracy_model(device)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=lr)
    
    # Initial evaluation
    init_acc, init_by_type = evaluate(model, val_loader, device)
    print(f"\nInitial val accuracy: {init_acc:.2%}")
    for t, a in init_by_type.items():
        print(f"  {t}: {a:.2%}")
    
    # Training
    model.train()
    step = 0
    best_val = 0.0
    
    pbar = tqdm(total=max_steps, desc="Training")
    
    start_time = time.time()
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            images = batch.images.to(device)
            questions = batch.questions
            answers = batch.answers
            
            if method == "supervised":
                loss = model.compute_supervised_loss(images, questions, answers)
            else:  # RL
                output = model(images, questions, mode="sample", temperature=1.0)
                
                rewards = torch.tensor([
                    1.0 if p.lower() == a.lower().strip() else 0.0
                    for p, a in zip(output.texts, answers)
                ], device=device)
                
                baseline = rewards.mean()
                advantages = rewards - baseline
                pg_loss = -(output.log_probs * advantages).mean()
                entropy_loss = -0.01 * output.entropy.mean()
                loss = pg_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
            optimizer.step()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Evaluate periodically
            if step % 200 == 0:
                val_acc, val_by_type = evaluate(model, val_loader, device)
                best_val = max(best_val, val_acc)
                pbar.set_postfix({"val_acc": f"{val_acc:.2%}"})
                model.train()
    
    pbar.close()
    
    elapsed = time.time() - start_time
    
    # Final evaluation
    test_acc, test_by_type = evaluate(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Best Val Accuracy: {best_val:.2%}")
    print(f"Training Time: {elapsed:.1f}s")
    print("\nAccuracy by Question Type:")
    for t, a in test_by_type.items():
        print(f"  {t}: {a:.2%}")
    print("=" * 60)
    
    # Save results
    os.makedirs("experiments/high_accuracy", exist_ok=True)
    with open("experiments/high_accuracy/results.json", "w") as f:
        json.dump({
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val,
            "by_type": test_by_type,
            "elapsed_seconds": elapsed,
            "method": method,
            "lr": lr,
            "max_steps": max_steps,
        }, f, indent=2)
    
    return test_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--method", default="supervised", choices=["supervised", "rl"])
    args = parser.parse_args()
    
    train(max_steps=args.steps, lr=args.lr, method=args.method)
