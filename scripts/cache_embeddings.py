#!/usr/bin/env python
"""
Pre-cache Image Embeddings

Pre-computes CLIP image embeddings and saves them to disk.
This allows ultra-fast training by loading embeddings instead of computing them.

Each experiment will take ~30-60 seconds instead of 10+ minutes.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from models.vision import create_vision_encoder


def cache_embeddings(
    data_dir: str = "data/generated",
    output_dir: str = "data/cached_embeddings",
    device: str = "cpu",
    batch_size: int = 32,
):
    """Pre-compute and cache CLIP embeddings for all images."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if GPU is available
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("CACHING IMAGE EMBEDDINGS")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)
    
    # Create vision encoder
    print("\nLoading CLIP model...")
    vision_encoder = create_vision_encoder(
        model_name="ViT-B-32",
        pretrained="openai",
        device=device,
        use_dummy=False,
    )
    vision_encoder.eval()
    
    preprocess = vision_encoder.get_preprocess()
    
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} split...")
        
        # Load metadata
        metadata_path = os.path.join(data_dir, split, "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"  Skipping {split} - no metadata found")
            continue
        
        with open(metadata_path, "r") as f:
            samples = json.load(f)
        
        embeddings = []
        sample_ids = []
        
        # Process in batches
        for i in tqdm(range(0, len(samples), batch_size), desc=f"  {split}"):
            batch_samples = samples[i:i+batch_size]
            batch_images = []
            
            for sample in batch_samples:
                image_path = sample["image_path"]
                
                # Handle paths
                if not os.path.isabs(image_path) and not os.path.exists(image_path):
                    if not image_path.startswith(data_dir):
                        image_path = os.path.join(data_dir, image_path)
                
                # Load and preprocess image
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image)
                batch_images.append(image_tensor)
                sample_ids.append(sample["sample_id"])
            
            # Stack and encode
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                batch_embeddings = vision_encoder(batch_tensor)
            
            embeddings.append(batch_embeddings.cpu())
        
        # Concatenate and save
        all_embeddings = torch.cat(embeddings, dim=0)
        
        output_path = os.path.join(output_dir, f"{split}_embeddings.pt")
        torch.save({
            "embeddings": all_embeddings,
            "sample_ids": sample_ids,
        }, output_path)
        
        print(f"  Saved {len(sample_ids)} embeddings to {output_path}")
        print(f"  Embedding shape: {all_embeddings.shape}")
    
    print("\n" + "=" * 60)
    print("CACHING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/generated")
    parser.add_argument("--output_dir", default="data/cached_embeddings")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    cache_embeddings(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
