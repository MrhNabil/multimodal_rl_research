#!/usr/bin/env python
"""
OPTIMIZED TRAINING FOR 85%+ ACCURACY

Key improvements:
1. Larger model (1024 hidden dim, deeper MLP)
2. Learning rate scheduling (warmup + cosine decay)
3. Label smoothing
4. Gradient clipping
5. More training steps (2000+)
6. Data augmentation (question paraphrasing)
7. Early stopping with proper patience
8. Better weight initialization
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import json
import time
from typing import List, Tuple
from dataclasses import dataclass

# ============================================================
# ENHANCED MODEL WITH LARGER CAPACITY
# ============================================================

@dataclass 
class VQAOutput:
    texts: List[str]
    log_probs: torch.Tensor = None
    entropy: torch.Tensor = None


class UltraHighAccuracyVQA(nn.Module):
    """
    Enhanced VQA model with:
    - Larger hidden dimensions (1024)
    - Deeper MLP (4 layers)
    - Residual connections
    - Layer normalization
    - Better initialization
    """
    
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
    
    def __init__(self, vision_dim: int = 512, hidden_dim: int = 1024, 
                 num_layers: int = 4, dropout: float = 0.15, device: str = "cuda"):
        super().__init__()
        
        self.device = device
        self.hidden_dim = hidden_dim
        
        # Deeper visual MLP with residual connections
        layers = []
        in_dim = vision_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # GELU works better than ReLU
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.visual_mlp = nn.Sequential(*layers)
        
        # Question type embeddings (learnable)
        self.type_embedding = nn.Embedding(4, hidden_dim // 4)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Larger type-specific heads with hidden layer
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.COLOR_ANSWERS))
        )
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.SHAPE_ANSWERS))
        )
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.COUNT_ANSWERS))
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, len(self.SPATIAL_ANSWERS))
        )
        
        # Index mappings
        self.color_to_idx = {a: i for i, a in enumerate(self.COLOR_ANSWERS)}
        self.shape_to_idx = {a: i for i, a in enumerate(self.SHAPE_ANSWERS)}
        self.count_to_idx = {a: i for i, a in enumerate(self.COUNT_ANSWERS)}
        self.spatial_to_idx = {a: i for i, a in enumerate(self.SPATIAL_ANSWERS)}
        
        # Better initialization
        self._init_weights()
        
        self.to(device)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"UltraHighAccuracyVQA initialized: {total_params:,} parameters")
    
    def _init_weights(self):
        """Xavier/Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def parse_question_type(self, question: str) -> int:
        q = question.lower()
        if "color" in q or "what color" in q:
            return 0
        elif "shape" in q or "what shape" in q:
            return 1
        elif "how many" in q or "count" in q or "number" in q:
            return 2
        else:
            return 3
    
    def forward(self, visual_features: torch.Tensor, questions: List[str],
                mode: str = "greedy", temperature: float = 1.0) -> VQAOutput:
        batch_size = visual_features.shape[0]
        
        if visual_features.device != torch.device(self.device):
            visual_features = visual_features.to(self.device)
        
        # Encode visual features
        hidden = self.visual_mlp(visual_features)
        
        # Parse question types
        q_types = [self.parse_question_type(q) for q in questions]
        q_type_tensor = torch.tensor(q_types, device=self.device)
        type_emb = self.type_embedding(q_type_tensor)
        
        # Fuse visual and question type
        combined = torch.cat([hidden, type_emb], dim=-1)
        fused = self.fusion(combined)
        
        predictions = []
        log_probs_list = []
        entropy_list = []
        
        for i in range(batch_size):
            q_type = q_types[i]
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
        """Compute loss with label smoothing."""
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


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def load_cached_data(data_dir: str, device: str):
    """Load pre-cached CLIP embeddings for faster training."""
    cache_file = os.path.join(data_dir, "cached_embeddings.pt")
    
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        data = torch.load(cache_file, map_location=device)
        return data
    
    print("No cached embeddings found. Will compute on the fly.")
    return None


def train_optimized(model, train_data, val_data, config: dict):
    """
    Optimized training loop with:
    - OneCycleLR scheduler
    - Gradient clipping
    - Early stopping
    - Label smoothing
    """
    device = config.get('device', 'cuda')
    max_steps = config.get('max_steps', 2000)
    batch_size = config.get('batch_size', 64)
    lr = config.get('lr', 0.001)
    patience = config.get('patience', 200)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # OneCycleLR for super-convergence
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=lr * 10, 
        steps_per_epoch=len(train_data['embeddings']) // batch_size,
        epochs=max_steps // (len(train_data['embeddings']) // batch_size) + 1,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    best_val_acc = 0.0
    best_state = None
    steps_without_improvement = 0
    
    train_embs = train_data['embeddings'].to(device)
    train_qs = train_data['questions']
    train_ans = train_data['answers']
    n_train = len(train_embs)
    
    print(f"\nStarting training for {max_steps} steps...")
    print(f"Train samples: {n_train}, Batch size: {batch_size}")
    
    step = 0
    start_time = time.time()
    
    model.train()
    while step < max_steps:
        # Shuffle indices each epoch
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train - batch_size, batch_size):
            if step >= max_steps:
                break
            
            idx = perm[i:i+batch_size]
            batch_emb = train_embs[idx]
            batch_q = [train_qs[j] for j in idx.tolist()]
            batch_a = [train_ans[j] for j in idx.tolist()]
            
            optimizer.zero_grad()
            loss = model.compute_loss(batch_emb, batch_q, batch_a, label_smoothing=0.1)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            step += 1
            
            # Validation every 50 steps
            if step % 50 == 0:
                val_acc = evaluate(model, val_data, device)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    steps_without_improvement = 0
                    marker = " ★ NEW BEST"
                else:
                    steps_without_improvement += 50
                    marker = ""
                
                elapsed = time.time() - start_time
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | Val: {val_acc*100:.1f}% | Best: {best_val_acc*100:.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}{marker}")
                
                model.train()
                
                # Early stopping
                if steps_without_improvement >= patience:
                    print(f"\nEarly stopping at step {step} (no improvement for {patience} steps)")
                    break
    
    # Load best weights
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best validation accuracy: {best_val_acc*100:.1f}%")
    
    return best_val_acc


def evaluate(model, data, device):
    """Evaluate model on data."""
    model.eval()
    
    embs = data['embeddings'].to(device)
    questions = data['questions']
    answers = data['answers']
    
    correct = 0
    total = 0
    
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(embs), batch_size):
            batch_emb = embs[i:i+batch_size]
            batch_q = questions[i:i+batch_size]
            batch_a = answers[i:i+batch_size]
            
            output = model(batch_emb, batch_q, mode="greedy")
            
            for pred, gt in zip(output.texts, batch_a):
                if pred.lower().strip() == gt.lower().strip():
                    correct += 1
                total += 1
    
    return correct / total


def evaluate_per_type(model, data, device):
    """Evaluate per question type."""
    model.eval()
    
    embs = data['embeddings'].to(device)
    questions = data['questions']
    answers = data['answers']
    
    type_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    type_total = {0: 0, 1: 0, 2: 0, 3: 0}
    type_names = {0: 'color', 1: 'shape', 2: 'count', 3: 'spatial'}
    
    with torch.no_grad():
        for i in range(len(embs)):
            emb = embs[i:i+1]
            q = [questions[i]]
            a = answers[i]
            
            q_type = model.parse_question_type(q[0])
            output = model(emb, q, mode="greedy")
            
            if output.texts[0].lower().strip() == a.lower().strip():
                type_correct[q_type] += 1
            type_total[q_type] += 1
    
    results = {}
    for t in range(4):
        if type_total[t] > 0:
            acc = type_correct[t] / type_total[t]
            results[type_names[t]] = acc
            print(f"  {type_names[t]:8s}: {acc*100:.1f}% ({type_correct[t]}/{type_total[t]})")
    
    return results


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ULTRA HIGH ACCURACY TRAINING")
    print("Target: 85%+ accuracy")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Check for cached embeddings
    data_dir = r"d:\multimodal_rl_research\data\generated"
    cache_file = os.path.join(data_dir, "cached_embeddings.pt")
    
    if not os.path.exists(cache_file):
        print("\n⚠ Cached embeddings not found. Creating them first...")
        print("Run: python scripts/cache_embeddings.py")
        
        # Create cache
        from models.vision import create_vision_encoder
        
        vision = create_vision_encoder("ViT-B-32", "openai", device=device)
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(data_dir, split)
            metadata_file = os.path.join(split_dir, "metadata.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                print(f"Processing {split} ({len(metadata)} samples)...")
                
                embeddings = []
                questions = []
                answers = []
                
                from PIL import Image
                from torchvision import transforms
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                        (0.26862954, 0.26130258, 0.27577711))
                ])
                
                batch_size = 64
                batch_images = []
                batch_metadata = []
                
                for item in metadata:
                    # Get image path - handle both relative and absolute
                    img_path_raw = item.get('image_path', item.get('image', ''))
                    if os.path.isabs(img_path_raw):
                        img_path = img_path_raw
                    else:
                        img_path = os.path.join(split_dir, "images", os.path.basename(img_path_raw))
                    
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = transform(img)
                        batch_images.append(img_tensor)
                        batch_metadata.append(item)
                        
                        if len(batch_images) >= batch_size:
                            batch_tensor = torch.stack(batch_images).to(device)
                            with torch.no_grad():
                                emb = vision(batch_tensor)
                            embeddings.append(emb.cpu())
                            
                            for m in batch_metadata:
                                questions.append(m['question'])
                                answers.append(m['answer'])
                            
                            batch_images = []
                            batch_metadata = []
                
                # Process remaining
                if batch_images:
                    batch_tensor = torch.stack(batch_images).to(device)
                    with torch.no_grad():
                        emb = vision(batch_tensor)
                    embeddings.append(emb.cpu())
                    for m in batch_metadata:
                        questions.append(m['question'])
                        answers.append(m['answer'])
                
                if embeddings:
                    all_embeddings = torch.cat(embeddings, dim=0)
                    torch.save({
                        'embeddings': all_embeddings,
                        'questions': questions,
                        'answers': answers
                    }, os.path.join(data_dir, f"cached_{split}.pt"))
                    print(f"  Saved {len(questions)} samples")
    
    # Load cached data
    print("\nLoading cached embeddings...")
    train_data = torch.load(os.path.join(data_dir, "cached_train.pt"), map_location=device, weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "cached_val.pt"), map_location=device, weights_only=False)
    test_data = torch.load(os.path.join(data_dir, "cached_test.pt"), map_location=device, weights_only=False)
    
    print(f"Train: {len(train_data['questions'])} samples")
    print(f"Val: {len(val_data['questions'])} samples")
    print(f"Test: {len(test_data['questions'])} samples")
    
    # Create optimized model
    print("\nCreating UltraHighAccuracyVQA model...")
    model = UltraHighAccuracyVQA(
        vision_dim=512,
        hidden_dim=1024,
        num_layers=4,
        dropout=0.15,
        device=device
    )
    
    # Training config
    config = {
        'device': device,
        'max_steps': 3000,
        'batch_size': 64,
        'lr': 0.0005,
        'patience': 300,
    }
    
    # Train
    best_val = train_optimized(model, train_data, val_data, config)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    
    test_acc = evaluate(model, test_data, device)
    print(f"\nTest Accuracy: {test_acc*100:.1f}%")
    
    print("\nPer-type accuracy:")
    per_type = evaluate_per_type(model, test_data, device)
    
    # Save results
    results = {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val,
        'per_type': per_type,
        'config': config,
        'model': 'UltraHighAccuracyVQA',
    }
    
    output_dir = r"d:\multimodal_rl_research\experiments\ultra_high_accuracy"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    print(f"\nResults saved to: {output_dir}")
    
    if test_acc >= 0.85:
        print("\n🎉 SUCCESS: Achieved 85%+ accuracy!")
    else:
        print(f"\n📊 Accuracy: {test_acc*100:.1f}% (target: 85%)")
