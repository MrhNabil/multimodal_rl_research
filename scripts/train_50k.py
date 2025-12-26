#!/usr/bin/env python
"""Train the proven model on 50K dataset."""

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

@dataclass 
class VQAOutput:
    texts: List[str]
    log_probs: torch.Tensor = None
    entropy: torch.Tensor = None


class ProvenVQAModel(nn.Module):
    """Proven model architecture that got 68.7% on 5K data."""
    
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
    
    def __init__(self, vision_dim: int = 512, hidden_dim: int = 512, device: str = "cuda"):
        super().__init__()
        
        self.device = device
        
        self.visual_mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.color_head = nn.Linear(hidden_dim, len(self.COLOR_ANSWERS))
        self.shape_head = nn.Linear(hidden_dim, len(self.SHAPE_ANSWERS))
        self.count_head = nn.Linear(hidden_dim, len(self.COUNT_ANSWERS))
        self.spatial_head = nn.Linear(hidden_dim, len(self.SPATIAL_ANSWERS))
        
        self.color_to_idx = {a: i for i, a in enumerate(self.COLOR_ANSWERS)}
        self.shape_to_idx = {a: i for i, a in enumerate(self.SHAPE_ANSWERS)}
        self.count_to_idx = {a: i for i, a in enumerate(self.COUNT_ANSWERS)}
        self.spatial_to_idx = {a: i for i, a in enumerate(self.SPATIAL_ANSWERS)}
        
        self.to(device)
        
        total = sum(p.numel() for p in self.parameters())
        print(f"ProvenVQAModel: {total:,} parameters")
    
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
        
        predictions = []
        for i, q_type in enumerate(q_types):
            h = hidden[i:i+1]
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
        
        total_loss = 0.0
        count = 0
        
        for i, (q, ans) in enumerate(zip(questions, answers)):
            q_type = self.parse_question_type(q)
            h = hidden[i:i+1]
            ans_lower = ans.lower().strip()
            
            if q_type == 0 and ans_lower in self.color_to_idx:
                logits = self.color_head(h)
                target = torch.tensor([self.color_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target)
                count += 1
            elif q_type == 1 and ans_lower in self.shape_to_idx:
                logits = self.shape_head(h)
                target = torch.tensor([self.shape_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target)
                count += 1
            elif q_type == 2 and ans_lower in self.count_to_idx:
                logits = self.count_head(h)
                target = torch.tensor([self.count_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target)
                count += 1
            elif q_type == 3 and ans_lower in self.spatial_to_idx:
                logits = self.spatial_head(h)
                target = torch.tensor([self.spatial_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target)
                count += 1
        
        return total_loss / max(count, 1)


def train_on_50k(data_dir: str, max_steps: int = 10000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("Loading 50K data...")
    train_data = torch.load(os.path.join(data_dir, "cached_train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "cached_val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(data_dir, "cached_test.pt"), weights_only=False)
    
    print(f"Train: {len(train_data['questions'])}")
    print(f"Val: {len(val_data['questions'])}")
    print(f"Test: {len(test_data['questions'])}")
    
    model = ProvenVQAModel(vision_dim=512, hidden_dim=512, device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    
    train_embs = train_data['embeddings'].to(device)
    train_qs = train_data['questions']
    train_ans = train_data['answers']
    n_train = len(train_embs)
    
    batch_size = 128
    best_val_acc = 0.0
    best_state = None
    
    print(f"\nTraining for {max_steps} steps...")
    start_time = time.time()
    
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
                    marker = " ★"
                else:
                    marker = ""
                
                elapsed = time.time() - start_time
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | Val: {val_acc*100:.1f}% | Best: {best_val_acc*100:.1f}%{marker} | Time: {elapsed:.0f}s")
                model.train()
    
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    
    # Final evaluation
    test_acc = evaluate(model, test_data, device)
    per_type = evaluate_per_type(model, test_data, device)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    print(f"Best Val: {best_val_acc*100:.1f}%")
    
    return test_acc, per_type, model


def evaluate(model, data, device):
    model.eval()
    embs = data['embeddings'].to(device)
    questions = data['questions']
    answers = data['answers']
    
    correct = 0
    with torch.no_grad():
        for i in range(0, len(embs), 256):
            batch_emb = embs[i:i+256]
            batch_q = questions[i:i+256]
            batch_a = answers[i:i+256]
            output = model(batch_emb, batch_q)
            for pred, gt in zip(output.texts, batch_a):
                if pred.lower().strip() == gt.lower().strip():
                    correct += 1
    return correct / len(embs)


def evaluate_per_type(model, data, device):
    model.eval()
    embs = data['embeddings'].to(device)
    questions = data['questions']
    answers = data['answers']
    
    type_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    type_total = {0: 0, 1: 0, 2: 0, 3: 0}
    type_names = {0: 'color', 1: 'shape', 2: 'count', 3: 'spatial'}
    
    with torch.no_grad():
        for i in range(len(embs)):
            q_type = model.parse_question_type(questions[i])
            output = model(embs[i:i+1], [questions[i]])
            if output.texts[0].lower().strip() == answers[i].lower().strip():
                type_correct[q_type] += 1
            type_total[q_type] += 1
    
    results = {}
    print("\nPer-type accuracy:")
    for t in range(4):
        if type_total[t] > 0:
            acc = type_correct[t] / type_total[t]
            results[type_names[t]] = acc
            print(f"  {type_names[t]:8s}: {acc*100:.1f}% ({type_correct[t]}/{type_total[t]})")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING ON 50K DATASET")
    print("=" * 60)
    
    test_acc, per_type, model = train_on_50k(
        data_dir="data/generated_50k",
        max_steps=10000
    )
    
    # Save results
    output_dir = r"d:\multimodal_rl_research\experiments\50k_training"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump({
            'test_accuracy': test_acc,
            'per_type': per_type,
            'data': '50K samples',
        }, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    if test_acc >= 0.85:
        print("\n🎉 SUCCESS: Achieved 85%+ accuracy!")
    else:
        print(f"\n📊 Final accuracy: {test_acc*100:.1f}%")
