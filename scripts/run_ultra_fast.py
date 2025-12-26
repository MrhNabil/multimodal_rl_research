#!/usr/bin/env python
"""
Ultra-Fast Experiment Runner

Uses MLP classifier instead of T5 for ~100x faster training.
Each experiment takes seconds instead of minutes.

Run all 61 experiments in ~30-60 minutes instead of 6+ hours.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.helpers import set_seed, load_config, merge_configs, ensure_dir
from data.dataset import MultimodalDataset, create_dataloader
from models.vision import create_vision_encoder
from models.projection import create_projection_layer
from models.fast_reasoning import FastVQAReasoner, FastMultimodalVQA


def create_fast_model(device: str = "cpu", use_dummy: bool = False):
    """Create the fast multimodal VQA model."""
    
    # Create vision encoder
    vision_encoder = create_vision_encoder(
        model_name="ViT-B-32",
        pretrained="openai",
        device=device,
        use_dummy=use_dummy,
    )
    
    vision_dim = vision_encoder.get_embedding_dim()
    
    # Create fast reasoner (MLP classifier)
    reasoner = FastVQAReasoner(
        input_dim=vision_dim,
        hidden_dim=256,
        num_layers=2,
        device=device,
    )
    
    # Create projection layer and move to device
    projection_layer = create_projection_layer(
        input_dim=vision_dim,
        output_dim=vision_dim,
        use_hidden=False,
    ).to(device)
    
    model = FastMultimodalVQA(
        vision_encoder=vision_encoder,
        reasoner=reasoner,
        projection_layer=projection_layer,
    )
    
    # Move entire model to device
    model = model.to(device)
    
    return model


def evaluate_model(model, dataloader, device="cpu") -> Dict:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    per_type_correct = {}
    per_type_total = {}
    
    with torch.no_grad():
        for batch in dataloader:
            # BatchedSample is a dataclass, access via attributes
            images = batch.images.to(device)
            questions = batch.questions
            answers = batch.answers
            q_types = batch.question_types if hasattr(batch, 'question_types') else ["unknown"] * len(questions)
            
            # Get predictions
            output = model(images, questions, mode="greedy")
            predictions = output.texts
            
            # Count accuracy
            for pred, target, q_type in zip(predictions, answers, q_types):
                pred_clean = pred.lower().strip()
                target_clean = target.lower().strip()
                
                if q_type not in per_type_correct:
                    per_type_correct[q_type] = 0
                    per_type_total[q_type] = 0
                
                per_type_total[q_type] += 1
                total += 1
                
                if pred_clean == target_clean:
                    correct += 1
                    per_type_correct[q_type] += 1
    
    accuracy = correct / total if total > 0 else 0
    per_type_accuracy = {
        q_type: per_type_correct[q_type] / per_type_total[q_type]
        for q_type in per_type_total
    }
    
    return {
        "accuracy": accuracy,
        "per_type_accuracy": per_type_accuracy,
        "correct": correct,
        "total": total,
    }


def train_supervised(
    model,
    train_loader,
    val_loader,
    config: Dict,
    device: str = "cpu",
) -> Dict:
    """Train with supervised learning (cross-entropy)."""
    
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=0.01,
    )
    
    max_steps = config.get("max_steps", 200)
    eval_every = config.get("eval_every", 50)
    
    model.train()
    step = 0
    best_val_acc = 0.0
    train_losses = []
    
    pbar = tqdm(total=max_steps, desc="Training")
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            # BatchedSample is a dataclass
            images = batch.images.to(device)
            questions = batch.questions
            answers = batch.answers
            
            # Forward pass
            loss = model.compute_supervised_loss(images, questions, answers)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            pbar.update(1)
            
            # Evaluate
            if step % eval_every == 0:
                val_metrics = evaluate_model(model, val_loader, device)
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                model.train()
    
    pbar.close()
    
    return {
        "final_train_loss": sum(train_losses[-10:]) / min(10, len(train_losses)),
        "best_val_accuracy": best_val_acc,
    }


def compute_reward(pred: str, target: str, reward_type: str) -> float:
    """Compute reward based on reward type."""
    pred_clean = pred.lower().strip()
    target_clean = target.lower().strip()
    
    if reward_type == "exact_match":
        return 1.0 if pred_clean == target_clean else 0.0
    
    elif reward_type == "partial_match":
        # Partial credit for partially matching answers
        if pred_clean == target_clean:
            return 1.0
        elif pred_clean in target_clean or target_clean in pred_clean:
            return 0.5
        else:
            return 0.0
    
    elif reward_type == "length_penalty":
        # Penalize longer wrong answers
        if pred_clean == target_clean:
            return 1.0
        else:
            return -0.1 * len(pred_clean) / 10.0
    
    elif reward_type == "progressive":
        # Progressive reward based on word overlap
        if pred_clean == target_clean:
            return 1.0
        pred_words = set(pred_clean.split())
        target_words = set(target_clean.split())
        overlap = len(pred_words & target_words)
        if overlap > 0:
            return 0.3 * overlap / max(len(target_words), 1)
        return 0.0
    
    else:  # Default to exact match
        return 1.0 if pred_clean == target_clean else 0.0


def train_rl(
    model,
    train_loader,
    val_loader,
    config: Dict,
    device: str = "cpu",
) -> Dict:
    """Train with REINFORCE (policy gradient)."""
    
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=0.01,
    )
    
    max_steps = config.get("max_steps", 200)
    eval_every = config.get("eval_every", 50)
    temperature = config.get("temperature", 1.0)
    entropy_coef = config.get("entropy_coef", 0.01)
    reward_type = config.get("reward_type", "exact_match")  # NEW: use reward_type from config
    
    # Moving average baseline for variance reduction
    baseline = 0.0
    baseline_decay = config.get("baseline_decay", 0.99)
    
    model.train()
    step = 0
    best_val_acc = 0.0
    rewards_history = []
    
    pbar = tqdm(total=max_steps, desc="RL Training")
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            # BatchedSample is a dataclass
            images = batch.images.to(device)
            questions = batch.questions
            answers = batch.answers
            
            # Sample actions
            output = model(images, questions, mode="sample", temperature=temperature)
            predictions = output.texts
            log_probs = output.log_probs  # [B, 1]
            entropy = output.entropy  # [B]
            
            # Compute rewards based on reward_type from config
            rewards = [
                compute_reward(pred, target, reward_type)
                for pred, target in zip(predictions, answers)
            ]
            
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            mean_reward = rewards.mean().item()
            rewards_history.append(mean_reward)
            
            # Update baseline
            baseline = baseline_decay * baseline + (1 - baseline_decay) * mean_reward
            
            # Compute advantage
            advantages = rewards - baseline
            
            # Policy gradient loss
            pg_loss = -(log_probs.squeeze(-1) * advantages).mean()
            
            # Entropy bonus
            entropy_loss = -entropy_coef * entropy.mean()
            
            # Total loss
            loss = pg_loss + entropy_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
            optimizer.step()
            
            step += 1
            
            pbar.set_postfix({
                "reward": f"{mean_reward:.3f}",
                "loss": f"{loss.item():.4f}"
            })
            pbar.update(1)
            
            # Evaluate
            if step % eval_every == 0:
                val_metrics = evaluate_model(model, val_loader, device)
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                model.train()
    
    pbar.close()
    
    return {
        "mean_reward": sum(rewards_history) / len(rewards_history),
        "best_val_accuracy": best_val_acc,
    }


def run_experiment(
    config_path: str,
    data_dir: str = "data/generated",
    output_dir: str = "experiments/results_fast",
    use_dummy: bool = False,
) -> Dict:
    """Run a single experiment with the fast model."""
    
    # Load config
    base_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "base_config.yaml"
    )
    
    base_config = load_config(base_config_path)
    exp_config = load_config(config_path)
    config = merge_configs(base_config, exp_config)
    
    # Extract experiment name
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name} (FAST MODE)")
    print(f"{'='*60}")
    
    # Create model
    model = create_fast_model(device=device, use_dummy=use_dummy)
    
    # Get question types filter from config (if specified)
    question_types_filter = config.get("data", {}).get("question_types", None)
    
    # Load data with optional question type filtering
    train_dataset = MultimodalDataset(data_dir=data_dir, split="train", question_types=question_types_filter)
    val_dataset = MultimodalDataset(data_dir=data_dir, split="val", question_types=question_types_filter)
    test_dataset = MultimodalDataset(data_dir=data_dir, split="test", question_types=question_types_filter)
    
    batch_size = config.get("training", {}).get("batch_size", 16)
    
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Training config - now includes reward_type!
    training_config = {
        "max_steps": config.get("training", {}).get("max_steps", 200),
        "eval_every": config.get("training", {}).get("eval_every", 50),
        "learning_rate": config.get("training", {}).get("optimizer", {}).get("learning_rate", 1e-4),
        "temperature": config.get("reinforce", {}).get("temperature", 1.0),
        "entropy_coef": config.get("reinforce", {}).get("entropy_coef", 0.01),
        "baseline_decay": config.get("reinforce", {}).get("baseline_decay", 0.99),
        "reward_type": config.get("reinforce", {}).get("reward_type", "exact_match"),  # NEW
    }
    
    method = config.get("training", {}).get("method", "rl")
    
    start_time = time.time()
    
    # Train
    if method == "frozen":
        # Just evaluate, no training
        train_results = {"best_val_accuracy": 0.0}
    elif method == "supervised":
        train_results = train_supervised(model, train_loader, val_loader, training_config, device)
    else:
        train_results = train_rl(model, train_loader, val_loader, training_config, device)
    
    # Final evaluation
    test_metrics = evaluate_model(model, test_loader, device)
    
    elapsed = time.time() - start_time
    
    # Save results
    exp_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    results = {
        "experiment_name": exp_name,
        "method": method,
        "accuracy": test_metrics["accuracy"],
        "per_type_accuracy": test_metrics["per_type_accuracy"],
        "best_val_accuracy": train_results.get("best_val_accuracy", 0.0),
        "elapsed_seconds": elapsed,
        "max_steps": training_config["max_steps"],
        "learning_rate": training_config["learning_rate"],
        "batch_size": batch_size,
        "seed": seed,
    }
    
    with open(os.path.join(exp_output_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ {exp_name}: accuracy={results['accuracy']:.4f}, time={elapsed:.1f}s")
    
    return results


def run_all_experiments(
    config_dir: str = "config/experiments",
    data_dir: str = "data/generated",
    output_dir: str = "experiments/results_fast",
    max_experiments: int = None,
    start_from: int = 1,
    use_dummy: bool = False,
):
    """Run all experiments with the fast model."""
    
    # Find all config files
    configs = sorted([
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.startswith("exp_") and f.endswith(".yaml")
    ])
    
    # Filter
    configs = configs[start_from - 1:]
    if max_experiments:
        configs = configs[:max_experiments]
    
    print("=" * 60)
    print("ULTRA-FAST EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Experiments to run: {len(configs)}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    all_results = []
    total_start = time.time()
    
    for i, config_path in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]", end="")
        
        try:
            results = run_experiment(
                config_path=config_path,
                data_dir=data_dir,
                output_dir=output_dir,
                use_dummy=use_dummy,
            )
            results["success"] = True
        except Exception as e:
            results = {
                "experiment_name": os.path.basename(config_path),
                "success": False,
                "error": str(e),
            }
            print(f"  ✗ Failed: {e}")
        
        all_results.append(results)
        
        # Save batch summary
        with open(os.path.join(output_dir, "batch_summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH COMPLETE!")
    print("=" * 60)
    
    successful = [r for r in all_results if r.get("success", False)]
    print(f"Successful: {len(successful)}/{len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if successful:
        sorted_results = sorted(successful, key=lambda x: x.get("accuracy", 0), reverse=True)
        print("\nTop 10 by Accuracy:")
        print("-" * 40)
        for r in sorted_results[:10]:
            print(f"  {r['experiment_name'][:30]}: {r.get('accuracy', 0):.4f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fast experiment runner")
    parser.add_argument("--config", type=str, help="Single config to run")
    parser.add_argument("--config_dir", type=str, default="config/experiments")
    parser.add_argument("--data_dir", type=str, default="data/generated")
    parser.add_argument("--output_dir", type=str, default="experiments/results_fast")
    parser.add_argument("--max_experiments", type=int, default=None)
    parser.add_argument("--start_from", type=int, default=1)
    parser.add_argument("--use_dummy", action="store_true")
    args = parser.parse_args()
    
    if args.config:
        run_experiment(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_dummy=args.use_dummy,
        )
    else:
        run_all_experiments(
            config_dir=args.config_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_experiments=args.max_experiments,
            start_from=args.start_from,
            use_dummy=args.use_dummy,
        )


if __name__ == "__main__":
    main()
