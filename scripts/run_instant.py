#!/usr/bin/env python
"""
INSTANT Experiment Runner - Uses Pre-cached Embeddings

Each experiment takes 30-60 seconds instead of 10+ minutes by
loading pre-computed CLIP embeddings from disk.

Run `python scripts/cache_embeddings.py` first to generate the cache.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.helpers import set_seed, load_config, merge_configs


# ============================================================================
# Cached Embedding Dataset (NO CLIP, just load tensors)
# ============================================================================

class CachedEmbeddingDataset(Dataset):
    """Dataset that loads pre-computed embeddings from disk."""
    
    def __init__(
        self,
        data_dir: str = "data/generated",
        cache_dir: str = "data/cached_embeddings",
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.split = split
        
        # Load metadata
        metadata_path = os.path.join(data_dir, split, "metadata.json")
        with open(metadata_path, "r") as f:
            self.samples = json.load(f)
        
        # Load cached embeddings
        cache_path = os.path.join(cache_dir, f"{split}_embeddings.pt")
        if os.path.exists(cache_path):
            cache_data = torch.load(cache_path)
            self.embeddings = cache_data["embeddings"]
            self.cached_ids = cache_data["sample_ids"]
            
            # Create ID to index mapping
            self.id_to_idx = {sid: i for i, sid in enumerate(self.cached_ids)}
            print(f"Loaded {len(self.cached_ids)} cached embeddings for {split}")
        else:
            raise FileNotFoundError(
                f"Cache not found at {cache_path}. "
                f"Run: python scripts/cache_embeddings.py"
            )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample["sample_id"]
        
        # Get cached embedding
        cache_idx = self.id_to_idx[sample_id]
        embedding = self.embeddings[cache_idx]
        
        return {
            "embedding": embedding,
            "question": sample["question"],
            "answer": sample["answer"],
            "question_type": sample["question_type"],
        }


def collate_cached(batch):
    """Collate function for cached embeddings."""
    embeddings = torch.stack([b["embedding"] for b in batch])
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]
    question_types = [b["question_type"] for b in batch]
    
    return {
        "embeddings": embeddings,
        "questions": questions,
        "answers": answers,
        "question_types": question_types,
    }


# ============================================================================
# Fast MLP Classifier (same as before but simplified)
# ============================================================================

class FastClassifier(nn.Module):
    """Ultra-fast MLP classifier for VQA."""
    
    ANSWERS = [
        "red", "blue", "green", "yellow",
        "cube", "sphere", "cylinder", 
        "0", "1", "2", "3", "4", "5",
        "left", "center", "right", "none",
        "unknown"
    ]
    
    def __init__(self, input_dim=512, hidden_dim=256, device="cpu"):
        super().__init__()
        
        self.device = device
        self.answer_to_idx = {a: i for i, a in enumerate(self.ANSWERS)}
        self.idx_to_answer = {i: a for i, a in enumerate(self.ANSWERS)}
        
        # Question embedding (hash-based)
        self.q_embed = nn.Embedding(1000, hidden_dim)
        
        # Visual projection
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(self.ANSWERS)),
        )
        
        self.to(device)
    
    def forward(self, embeddings, questions):
        """Forward pass returning logits."""
        # Visual projection
        v = self.v_proj(embeddings.to(self.device))
        
        # Question encoding (hash)
        q_idx = torch.tensor([hash(q) % 1000 for q in questions], device=self.device)
        q = self.q_embed(q_idx)
        
        # Combine and classify
        combined = torch.cat([v, q], dim=-1)
        logits = self.classifier(combined)
        
        return logits
    
    def predict(self, embeddings, questions):
        """Get predictions."""
        logits = self.forward(embeddings, questions)
        pred_idx = logits.argmax(dim=-1)
        return [self.idx_to_answer[i.item()] for i in pred_idx]
    
    def sample(self, embeddings, questions, temperature=1.0):
        """Sample from policy (for RL)."""
        logits = self.forward(embeddings, questions) / temperature
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        sampled_log_probs = log_probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        answers = [self.idx_to_answer[i.item()] for i in sampled]
        
        return answers, sampled_log_probs, entropy
    
    def get_target_idx(self, answers):
        """Convert answer strings to indices."""
        return torch.tensor([
            self.answer_to_idx.get(a.lower().strip(), self.answer_to_idx["unknown"])
            for a in answers
        ], device=self.device)


# ============================================================================
# Training Functions
# ============================================================================

def train_supervised(model, train_loader, val_loader, config, device):
    """Supervised training."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    model.train()
    best_acc = 0.0
    
    pbar = tqdm(total=config["max_steps"], desc="Supervised")
    step = 0
    
    while step < config["max_steps"]:
        for batch in train_loader:
            if step >= config["max_steps"]:
                break
            
            embeddings = batch["embeddings"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]
            
            logits = model(embeddings, questions)
            targets = model.get_target_idx(answers)
            loss = F.cross_entropy(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if step % config["eval_every"] == 0:
                acc = evaluate(model, val_loader, device)
                best_acc = max(best_acc, acc)
                model.train()
    
    pbar.close()
    return {"best_val_accuracy": best_acc}


def train_rl(model, train_loader, val_loader, config, device):
    """RL (REINFORCE) training."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    model.train()
    best_acc = 0.0
    baseline = 0.0
    
    pbar = tqdm(total=config["max_steps"], desc="RL Training")
    step = 0
    
    while step < config["max_steps"]:
        for batch in train_loader:
            if step >= config["max_steps"]:
                break
            
            embeddings = batch["embeddings"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]
            
            # Sample actions
            predictions, log_probs, entropy = model.sample(
                embeddings, questions, 
                temperature=config.get("temperature", 1.0)
            )
            
            # Compute rewards
            rewards = torch.tensor([
                1.0 if p.lower() == a.lower().strip() else 0.0
                for p, a in zip(predictions, answers)
            ], device=device)
            
            mean_reward = rewards.mean().item()
            baseline = 0.99 * baseline + 0.01 * mean_reward
            
            # Policy gradient
            advantages = rewards - baseline
            pg_loss = -(log_probs * advantages).mean()
            entropy_loss = -config.get("entropy_coef", 0.01) * entropy.mean()
            loss = pg_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({"reward": f"{mean_reward:.3f}"})
            
            if step % config["eval_every"] == 0:
                acc = evaluate(model, val_loader, device)
                best_acc = max(best_acc, acc)
                model.train()
    
    pbar.close()
    return {"best_val_accuracy": best_acc, "mean_reward": baseline}


def evaluate(model, dataloader, device):
    """Evaluate accuracy."""
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch["embeddings"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]
            
            predictions = model.predict(embeddings, questions)
            
            for pred, target in zip(predictions, answers):
                if pred.lower() == target.lower().strip():
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0


# ============================================================================
# Main Runner
# ============================================================================

def run_experiment(config_path, data_dir, cache_dir, output_dir):
    """Run a single experiment using cached embeddings."""
    
    # Load config
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "base_config.yaml")
    config = merge_configs(load_config(base_path), load_config(config_path))
    
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    set_seed(config.get("seed", 42))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*50}")
    print(f"INSTANT: {exp_name}")
    print(f"Device: {device}")
    print(f"{'='*50}")
    
    # Load datasets
    train_ds = CachedEmbeddingDataset(data_dir, cache_dir, "train")
    val_ds = CachedEmbeddingDataset(data_dir, cache_dir, "val")
    test_ds = CachedEmbeddingDataset(data_dir, cache_dir, "test")
    
    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_cached)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_cached)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_cached)
    
    # Create model
    model = FastClassifier(input_dim=512, hidden_dim=256, device=device)
    
    # Training config
    train_config = {
        "max_steps": config.get("training", {}).get("max_steps", 200),
        "eval_every": config.get("training", {}).get("eval_every", 50),
        "learning_rate": config.get("training", {}).get("optimizer", {}).get("learning_rate", 1e-4),
        "temperature": config.get("reinforce", {}).get("temperature", 1.0),
        "entropy_coef": config.get("reinforce", {}).get("entropy_coef", 0.01),
    }
    
    method = config.get("training", {}).get("method", "rl")
    
    start_time = time.time()
    
    if method == "frozen":
        train_results = {"best_val_accuracy": 0.0}
    elif method == "supervised":
        train_results = train_supervised(model, train_loader, val_loader, train_config, device)
    else:
        train_results = train_rl(model, train_loader, val_loader, train_config, device)
    
    # Final evaluation
    test_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time
    
    # Save results
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    results = {
        "experiment_name": exp_name,
        "method": method,
        "accuracy": test_acc,
        "best_val_accuracy": train_results.get("best_val_accuracy", 0),
        "elapsed_seconds": elapsed,
        "device": device,
    }
    
    with open(os.path.join(exp_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ {exp_name}: accuracy={test_acc:.4f}, time={elapsed:.1f}s")
    
    return results


def run_all(config_dir, data_dir, cache_dir, output_dir, max_exp=None, start=1):
    """Run all experiments."""
    
    configs = sorted([
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.startswith("exp_") and f.endswith(".yaml")
    ])[start-1:]
    
    if max_exp:
        configs = configs[:max_exp]
    
    print("=" * 60)
    print("INSTANT EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Running {len(configs)} experiments")
    print("=" * 60)
    
    results = []
    total_start = time.time()
    
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]", end="")
        try:
            r = run_experiment(cfg, data_dir, cache_dir, output_dir)
            r["success"] = True
            results.append(r)
        except Exception as e:
            print(f" ✗ {e}")
            results.append({"experiment": cfg, "success": False, "error": str(e)})
        
        with open(os.path.join(output_dir, "batch_summary.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print(f"COMPLETE! {len([r for r in results if r.get('success')])} succeeded")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    successful = sorted([r for r in results if r.get("success")], 
                       key=lambda x: x.get("accuracy", 0), reverse=True)
    if successful:
        print("\nTop 5:")
        for r in successful[:5]:
            print(f"  {r['experiment_name']}: {r['accuracy']:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Single config")
    parser.add_argument("--config_dir", default="config/experiments")
    parser.add_argument("--data_dir", default="data/generated")
    parser.add_argument("--cache_dir", default="data/cached_embeddings")
    parser.add_argument("--output_dir", default="experiments/results_instant")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()
    
    if args.config:
        run_experiment(args.config, args.data_dir, args.cache_dir, args.output_dir)
    else:
        run_all(args.config_dir, args.data_dir, args.cache_dir, args.output_dir, args.max, args.start)


if __name__ == "__main__":
    main()
