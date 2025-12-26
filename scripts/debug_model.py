#!/usr/bin/env python
"""Debug script to understand why accuracy is low."""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from collections import Counter

from data.dataset import MultimodalDataset, create_dataloader
from models.vision import create_vision_encoder
from models.projection import create_projection_layer
from models.fast_reasoning import FastVQAReasoner, FastMultimodalVQA

def debug_predictions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Check vocabulary
    print(f"\n=== Model Vocabulary ({len(FastVQAReasoner.ANSWER_VOCAB)} answers) ===")
    for i, ans in enumerate(FastVQAReasoner.ANSWER_VOCAB):
        print(f"  {i}: '{ans}'")
    
    # Load dataset and check answers
    print("\n=== Dataset Answer Distribution ===")
    test_ds = MultimodalDataset("data/generated", "test")
    
    dataset_answers = Counter()
    for i in range(len(test_ds)):
        sample = test_ds[i]
        dataset_answers[sample["answer"]] += 1
    
    print("Dataset answers:")
    for ans, count in dataset_answers.most_common():
        in_vocab = ans in FastVQAReasoner.ANSWER_VOCAB
        print(f"  '{ans}': {count} {'✓' if in_vocab else '✗ NOT IN VOCAB!'}")
    
    # Check for mismatches
    missing = set(dataset_answers.keys()) - set(FastVQAReasoner.ANSWER_VOCAB)
    if missing:
        print(f"\n⚠️  MISSING FROM VOCAB: {missing}")
    
    # Now test predictions on a few samples
    print("\n=== Testing Model Predictions ===")
    
    # Create model
    vision_encoder = create_vision_encoder("ViT-B-32", "openai", device=device)
    vision_dim = vision_encoder.get_embedding_dim()
    reasoner = FastVQAReasoner(input_dim=vision_dim, hidden_dim=256, num_layers=2, device=device)
    projection_layer = create_projection_layer(vision_dim, vision_dim, use_hidden=False).to(device)
    model = FastMultimodalVQA(vision_encoder, reasoner, projection_layer).to(device)
    
    # Get a batch
    loader = create_dataloader(test_ds, batch_size=10, shuffle=False)
    batch = next(iter(loader))
    
    with torch.no_grad():
        images = batch.images.to(device)
        output = model(images, batch.questions, mode="greedy")
    
    print("\nSample predictions (BEFORE training):")
    print("-" * 80)
    for i in range(min(10, len(batch.questions))):
        q = batch.questions[i]
        gt = batch.answers[i]
        pred = output.texts[i]
        match = "✓" if pred.lower() == gt.lower() else "✗"
        print(f"{match} Q: {q[:50]}")
        print(f"   GT: '{gt}' | Pred: '{pred}'")
        print()

if __name__ == "__main__":
    debug_predictions()
