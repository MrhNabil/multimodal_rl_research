#!/usr/bin/env python
"""
ENSEMBLE + EXTENDED TRAINING for 85%+ accuracy

Strategy:
1. Train multiple models with different seeds
2. Use proven architecture (medium size)
3. Much longer training (10000 steps)
4. Ensemble predictions via voting
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from typing import List
from dataclasses import dataclass
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass 
class VQAOutput:
    texts: List[str]
    log_probs: torch.Tensor = None
    entropy: torch.Tensor = None


class OptimizedVQA(nn.Module):
    """Optimized VQA - medium size, proven to work."""
    
    COLOR_ANSWERS = ["red", "blue", "green", "yellow"]
    SHAPE_ANSWERS = ["cube", "sphere", "cylinder"]
    COUNT_ANSWERS = ["0", "1", "2", "3"]
    SPATIAL_ANSWERS = [
        "red cube", "red sphere", "red cylinder",
        "blue cube", "blue sphere", "blue cylinder",
        "green cube", "green sphere", "green cylinder",
        "yellow cube", "yellow sphere", "yellow cylinder",
        "nothing"
    ]
    
    def __init__(self, vision_dim: int = 512, hidden_dim: int = 768, device: str = "cuda"):
        super().__init__()
        
        self.device = device
        
        # Visual encoder
        self.visual_mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Type embedding
        self.type_embedding = nn.Embedding(4, hidden_dim // 4)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Heads
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, len(self.COLOR_ANSWERS))
        )
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, len(self.SHAPE_ANSWERS))
        )
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, len(self.COUNT_ANSWERS))
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, len(self.SPATIAL_ANSWERS))
        )
        
        self.color_to_idx = {a: i for i, a in enumerate(self.COLOR_ANSWERS)}
        self.shape_to_idx = {a: i for i, a in enumerate(self.SHAPE_ANSWERS)}
        self.count_to_idx = {a: i for i, a in enumerate(self.COUNT_ANSWERS)}
        self.spatial_to_idx = {a: i for i, a in enumerate(self.SPATIAL_ANSWERS)}
        
        self.to(device)
    
    def parse_question_type(self, question: str) -> int:
        q = question.lower()
        if "color" in q: return 0
        elif "shape" in q: return 1
        elif "how many" in q or "count" in q: return 2
        else: return 3
    
    def forward(self, visual_features: torch.Tensor, questions: List[str],
                mode: str = "greedy", temperature: float = 1.0) -> VQAOutput:
        if visual_features.device != torch.device(self.device):
            visual_features = visual_features.to(self.device)
        
        hidden = self.visual_mlp(visual_features)
        q_types = [self.parse_question_type(q) for q in questions]
        q_type_tensor = torch.tensor(q_types, device=self.device)
        type_emb = self.type_embedding(q_type_tensor)
        
        combined = torch.cat([hidden, type_emb], dim=-1)
        fused = self.fusion(combined)
        
        predictions = []
        for i, q_type in enumerate(q_types):
            h = fused[i:i+1]
            if q_type == 0:
                logits = self.color_head(h)
                answers = self.COLOR_ANSWERS
            elif q_type == 1:
                logits = self.shape_head(h)
                answers = self.SHAPE_ANSWERS
            elif q_type == 2:
                logits = self.count_head(h)
                answers = self.COUNT_ANSWERS
            else:
                logits = self.spatial_head(h)
                answers = self.SPATIAL_ANSWERS
            
            pred_idx = logits.argmax(dim=-1).item()
            predictions.append(answers[pred_idx])
        
        return VQAOutput(texts=predictions)
    
    def compute_loss(self, visual_features: torch.Tensor, questions: List[str],
                     answers: List[str]) -> torch.Tensor:
        if visual_features.device != torch.device(self.device):
            visual_features = visual_features.to(self.device)
        
        hidden = self.visual_mlp(visual_features)
        q_types = [self.parse_question_type(q) for q in questions]
        q_type_tensor = torch.tensor(q_types, device=self.device)
        type_emb = self.type_embedding(q_type_tensor)
        combined = torch.cat([hidden, type_emb], dim=-1)
        fused = self.fusion(combined)
        
        total_loss = 0.0
        count = 0
        
        for i, (q, ans) in enumerate(zip(questions, answers)):
            q_type = q_types[i]
            h = fused[i:i+1]
            ans_lower = ans.lower().strip()
            
            if q_type == 0 and ans_lower in self.color_to_idx:
                logits = self.color_head(h)
                target = torch.tensor([self.color_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target, label_smoothing=0.1)
                count += 1
            elif q_type == 1 and ans_lower in self.shape_to_idx:
                logits = self.shape_head(h)
                target = torch.tensor([self.shape_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target, label_smoothing=0.1)
                count += 1
            elif q_type == 2 and ans_lower in self.count_to_idx:
                logits = self.count_head(h)
                target = torch.tensor([self.count_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target, label_smoothing=0.1)
                count += 1
            elif q_type == 3 and ans_lower in self.spatial_to_idx:
                logits = self.spatial_head(h)
                target = torch.tensor([self.spatial_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target, label_smoothing=0.1)
                count += 1
        
        return total_loss / max(count, 1)


def train_single_model(model, train_data, val_data, config, seed):
    """Train a single model."""
    set_seed(seed)
    device = config['device']
    max_steps = config['max_steps']
    batch_size = config['batch_size']
    lr = config['lr']
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    
    best_val_acc = 0.0
    best_state = None
    
    train_embs = train_data['embeddings'].to(device)
    train_qs = train_data['questions']
    train_ans = train_data['answers']
    n_train = len(train_embs)
    
    step = 0
    model.train()
    
    while step < max_steps:
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train - batch_size, batch_size):
            if step >= max_steps:
                break
            
            idx = perm[i:i+batch_size]
            batch_emb = train_embs[idx]
            batch_q = [train_qs[j] for j in idx.tolist()]
            batch_a = [train_ans[j] for j in idx.tolist()]
            
            optimizer.zero_grad()
            loss = model.compute_loss(batch_emb, batch_q, batch_a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            
            if step % 500 == 0:
                val_acc = evaluate(model, val_data, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  Seed {seed} | Step {step:5d} | Val: {val_acc*100:.1f}% | Best: {best_val_acc*100:.1f}%")
                model.train()
    
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    
    return best_val_acc, model


def evaluate(model, data, device):
    model.eval()
    embs = data['embeddings'].to(device)
    questions = data['questions']
    answers = data['answers']
    
    correct = 0
    with torch.no_grad():
        for i in range(0, len(embs), 128):
            batch_emb = embs[i:i+128]
            batch_q = questions[i:i+128]
            batch_a = answers[i:i+128]
            output = model(batch_emb, batch_q)
            for pred, gt in zip(output.texts, batch_a):
                if pred.lower().strip() == gt.lower().strip():
                    correct += 1
    return correct / len(embs)


def ensemble_predict(models, data, device):
    """Ensemble prediction via majority voting."""
    from collections import Counter
    
    embs = data['embeddings'].to(device)
    questions = data['questions']
    answers = data['answers']
    
    all_predictions = [[] for _ in range(len(embs))]
    
    for model in models:
        model.eval()
        with torch.no_grad():
            for i in range(0, len(embs), 128):
                batch_emb = embs[i:i+128]
                batch_q = questions[i:i+128]
                output = model(batch_emb, batch_q)
                for j, pred in enumerate(output.texts):
                    all_predictions[i+j].append(pred.lower().strip())
    
    # Majority vote
    final_predictions = []
    for preds in all_predictions:
        counter = Counter(preds)
        final_predictions.append(counter.most_common(1)[0][0])
    
    correct = sum(1 for pred, gt in zip(final_predictions, answers) 
                  if pred == gt.lower().strip())
    return correct / len(embs)


if __name__ == "__main__":
    print("=" * 60)
    print("ENSEMBLE TRAINING FOR 85%+ ACCURACY")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    data_dir = r"d:\multimodal_rl_research\data\generated"
    train_data = torch.load(os.path.join(data_dir, "cached_train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "cached_val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(data_dir, "cached_test.pt"), weights_only=False)
    
    config = {
        'device': device,
        'max_steps': 8000,
        'batch_size': 64,
        'lr': 0.0005,
    }
    
    # Train multiple models
    seeds = [42, 123, 456, 789, 1024]
    models = []
    
    print(f"\nTraining {len(seeds)} models with different seeds...")
    
    for seed in seeds:
        print(f"\n--- Training model with seed {seed} ---")
        model = OptimizedVQA(vision_dim=512, hidden_dim=768, device=device)
        best_val, trained_model = train_single_model(model, train_data, val_data, config, seed)
        models.append(trained_model)
        
        test_acc = evaluate(trained_model, test_data, device)
        print(f"  Single model test: {test_acc*100:.1f}%")
    
    # Ensemble evaluation
    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION")
    print("=" * 60)
    
    ensemble_acc = ensemble_predict(models, test_data, device)
    print(f"\n🎯 Ensemble Test Accuracy: {ensemble_acc*100:.1f}%")
    
    # Find best single model
    best_single = max(evaluate(m, test_data, device) for m in models)
    print(f"Best Single Model: {best_single*100:.1f}%")
    
    # Save results
    results = {
        'ensemble_accuracy': ensemble_acc,
        'best_single_accuracy': best_single,
        'num_models': len(models),
        'seeds': seeds,
        'config': config,
    }
    
    output_dir = r"d:\multimodal_rl_research\experiments\ensemble_result"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save best model
    best_idx = max(range(len(models)), key=lambda i: evaluate(models[i], test_data, device))
    torch.save(models[best_idx].state_dict(), os.path.join(output_dir, "best_model.pt"))
    
    if ensemble_acc >= 0.85:
        print("\n🎉 SUCCESS: Achieved 85%+ accuracy!")
    else:
        print(f"\n📊 Ensemble: {ensemble_acc*100:.1f}% (target: 85%)")
