#!/usr/bin/env python
"""Cache CLIP embeddings for the 50K dataset."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def cache_embeddings(data_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    from models.vision import create_vision_encoder
    vision = create_vision_encoder("ViT-B-32", "openai", device=device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711))
    ])
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        metadata_file = os.path.join(split_dir, "metadata.json")
        
        if not os.path.exists(metadata_file):
            print(f"Skipping {split} - no metadata")
            continue
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        print(f"\nProcessing {split} ({len(metadata)} samples)...")
        
        embeddings = []
        questions = []
        answers = []
        
        batch_size = 128
        batch_images = []
        batch_metadata = []
        
        for item in tqdm(metadata, desc=f"Caching {split}"):
            img_path_raw = item.get('image_path', '')
            if os.path.isabs(img_path_raw):
                img_path = img_path_raw
            else:
                # Try different path combinations
                img_path = os.path.join(split_dir, "images", os.path.basename(img_path_raw))
                if not os.path.exists(img_path):
                    img_path = img_path_raw
            
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
            output_file = os.path.join(data_dir, f"cached_{split}.pt")
            torch.save({
                'embeddings': all_embeddings,
                'questions': questions,
                'answers': answers
            }, output_file)
            print(f"  Saved {len(questions)} samples to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/generated_50k")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CACHING CLIP EMBEDDINGS")
    print("=" * 60)
    
    cache_embeddings(args.data_dir)
    
    print("\n✓ Done!")
