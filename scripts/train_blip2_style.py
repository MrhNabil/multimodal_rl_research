#!/usr/bin/env python
"""
BLIP-2 Style VQA Model and Training

BLIP-2 uses a Q-Former (Querying Transformer) to bridge frozen vision and language models.
Key innovation: Learnable query tokens that extract visual features relevant to the task.

This implementation:
1. Frozen CLIP visual encoder
2. Q-Former style cross-attention bridge
3. Type-specific answer heads
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
import math

@dataclass 
class VQAOutput:
    texts: List[str]
    log_probs: torch.Tensor = None
    entropy: torch.Tensor = None


class QFormerLayer(nn.Module):
    """Single Q-Former layer with self-attention and cross-attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention for query tokens
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention to visual features
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, queries: torch.Tensor, visual_features: torch.Tensor):
        # Self-attention
        x = queries
        attn_out, _ = self.self_attn(x, x, x)
        x = self.self_attn_norm(x + attn_out)
        
        # Cross-attention to visual features
        cross_out, _ = self.cross_attn(x, visual_features, visual_features)
        x = self.cross_attn_norm(x + cross_out)
        
        # FFN
        x = self.ffn_norm(x + self.ffn(x))
        
        return x


class QFormer(nn.Module):
    """Q-Former: Querying Transformer for vision-language bridging."""
    
    def __init__(self, hidden_dim: int = 768, num_queries: int = 32, 
                 num_layers: int = 6, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        
        # Learnable query tokens (key innovation of BLIP-2)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        
        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [B, D] visual features from CLIP
            
        Returns:
            [B, hidden_dim] pooled features
        """
        batch_size = visual_features.shape[0]
        
        # Expand visual features to sequence: [B, 1, D]
        vis_seq = visual_features.unsqueeze(1)
        
        # Expand query tokens for batch: [B, num_queries, hidden_dim]
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        # Apply Q-Former layers
        for layer in self.layers:
            queries = layer(queries, vis_seq)
        
        # Pool query outputs (mean pooling)
        pooled = queries.mean(dim=1)  # [B, hidden_dim]
        
        return self.output_proj(pooled)


class BLIP2StyleVQA(nn.Module):
    """BLIP-2 style VQA with Q-Former bridging."""
    
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
    
    def __init__(self, vision_dim: int = 512, hidden_dim: int = 768,
                 num_queries: int = 32, num_layers: int = 6, device: str = "cuda"):
        super().__init__()
        
        self.device = device
        self.hidden_dim = hidden_dim
        
        # Vision projection to match Q-Former dim
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        # Q-Former (the key component)
        self.qformer = QFormer(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=12,
            dropout=0.1
        )
        
        # Question type embedding
        self.type_embedding = nn.Embedding(4, hidden_dim // 4)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Type-specific output heads
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(self.COLOR_ANSWERS))
        )
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(self.SHAPE_ANSWERS))
        )
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(self.COUNT_ANSWERS))
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(self.SPATIAL_ANSWERS))
        )
        
        # Index mappings
        self.color_to_idx = {a: i for i, a in enumerate(self.COLOR_ANSWERS)}
        self.shape_to_idx = {a: i for i, a in enumerate(self.SHAPE_ANSWERS)}
        self.count_to_idx = {a: i for i, a in enumerate(self.COUNT_ANSWERS)}
        self.spatial_to_idx = {a: i for i, a in enumerate(self.SPATIAL_ANSWERS)}
        
        self._init_weights()
        self.to(device)
        
        total = sum(p.numel() for p in self.parameters())
        print(f"BLIP2StyleVQA initialized: {total:,} parameters")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
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
        
        # Project vision features
        vis_proj = self.vision_proj(visual_features)
        
        # Apply Q-Former
        qformer_out = self.qformer(vis_proj)
        
        # Get question type embeddings
        q_types = [self.parse_question_type(q) for q in questions]
        q_type_tensor = torch.tensor(q_types, device=self.device)
        type_emb = self.type_embedding(q_type_tensor)
        
        # Fuse Q-Former output with question type
        combined = torch.cat([qformer_out, type_emb], dim=-1)
        fused = self.fusion(combined)
        
        predictions = []
        log_probs_list = []
        entropy_list = []
        
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
            
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            log_probs_all = F.log_softmax(scaled_logits, dim=-1)
            
            if mode == "greedy":
                pred_idx = logits.argmax(dim=-1).item()
            else:
                pred_idx = torch.multinomial(probs, 1).item()
            
            predictions.append(answers[pred_idx])
            log_probs_list.append(log_probs_all[0, pred_idx])
            entropy_list.append(-(probs * log_probs_all).sum())
        
        log_probs = torch.stack(log_probs_list)
        entropy = torch.stack(entropy_list)
        
        return VQAOutput(texts=predictions, log_probs=log_probs, entropy=entropy)
    
    def compute_loss(self, visual_features: torch.Tensor, questions: List[str],
                     answers: List[str], label_smoothing: float = 0.1) -> torch.Tensor:
        if visual_features.device != torch.device(self.device):
            visual_features = visual_features.to(self.device)
        
        vis_proj = self.vision_proj(visual_features)
        qformer_out = self.qformer(vis_proj)
        
        q_types = [self.parse_question_type(q) for q in questions]
        q_type_tensor = torch.tensor(q_types, device=self.device)
        type_emb = self.type_embedding(q_type_tensor)
        
        combined = torch.cat([qformer_out, type_emb], dim=-1)
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
                total_loss += F.cross_entropy(logits, target, label_smoothing=label_smoothing)
                count += 1
            elif q_type == 1 and ans_lower in self.shape_to_idx:
                logits = self.shape_head(h)
                target = torch.tensor([self.shape_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target, label_smoothing=label_smoothing)
                count += 1
            elif q_type == 2 and ans_lower in self.count_to_idx:
                logits = self.count_head(h)
                target = torch.tensor([self.count_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target, label_smoothing=label_smoothing)
                count += 1
            elif q_type == 3 and ans_lower in self.spatial_to_idx:
                logits = self.spatial_head(h)
                target = torch.tensor([self.spatial_to_idx[ans_lower]], device=self.device)
                total_loss += F.cross_entropy(logits, target, label_smoothing=label_smoothing)
                count += 1
        
        return total_loss / max(count, 1)


def train_blip2_style(data_dir: str, max_steps: int = 5000, batch_size: int = 64, lr: float = 0.0005):
    """Train BLIP-2 style model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    print("Loading data...")
    train_data = torch.load(os.path.join(data_dir, "cached_train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "cached_val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(data_dir, "cached_test.pt"), weights_only=False)
    
    print(f"Train: {len(train_data['questions'])}")
    print(f"Val: {len(val_data['questions'])}")
    print(f"Test: {len(test_data['questions'])}")
    
    # Create model
    print("\nCreating BLIP2StyleVQA model...")
    model = BLIP2StyleVQA(
        vision_dim=512,
        hidden_dim=768,
        num_queries=32,
        num_layers=6,
        device=device
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    
    train_embs = train_data['embeddings'].to(device)
    train_qs = train_data['questions']
    train_ans = train_data['answers']
    n_train = len(train_embs)
    
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
            
            if step % 200 == 0:
                val_acc = evaluate(model, val_data, device)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    marker = " ★"
                else:
                    marker = ""
                
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | Val: {val_acc*100:.1f}% | Best: {best_val_acc*100:.1f}%{marker}")
                model.train()
    
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    
    elapsed = time.time() - start_time
    
    # Final evaluation
    test_acc = evaluate(model, test_data, device)
    per_type = evaluate_per_type(model, test_data, device)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    print(f"Best Val: {best_val_acc*100:.1f}%")
    print(f"Time: {elapsed:.1f}s")
    
    return test_acc, per_type, model


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
            print(f"  {type_names[t]:8s}: {acc*100:.1f}%")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/generated")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()
    
    print("=" * 60)
    print("BLIP-2 STYLE VQA TRAINING")
    print("=" * 60)
    
    test_acc, per_type, model = train_blip2_style(
        data_dir=args.data_dir,
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # Save results
    output_dir = r"d:\multimodal_rl_research\experiments\blip2_style"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump({
            'test_accuracy': test_acc,
            'per_type': per_type,
            'model': 'BLIP2StyleVQA',
        }, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    if test_acc >= 0.85:
        print("\n🎉 SUCCESS: Achieved 85%+ accuracy!")
    else:
        print(f"\n📊 Final accuracy: {test_acc*100:.1f}%")
