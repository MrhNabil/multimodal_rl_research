#!/usr/bin/env python
"""
Download Pretrained Models

Downloads and caches CLIP and T5 models for offline use.
Run this once before training to ensure all models are available.
"""

import os
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_clip(cache_dir: str):
    """Download CLIP model."""
    print("Downloading CLIP ViT-B/32...")
    
    try:
        import open_clip
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            cache_dir=cache_dir,
        )
        
        print(f"  ✓ CLIP model downloaded successfully")
        print(f"  Cache location: {cache_dir}")
        
    except Exception as e:
        print(f"  ✗ Failed to download CLIP: {e}")
        return False
    
    return True


def download_t5(cache_dir: str, model_name: str = "t5-small"):
    """Download T5 model."""
    print(f"Downloading {model_name}...")
    
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            legacy=False,
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        
        print(f"  ✓ {model_name} downloaded successfully")
        print(f"  Cache location: {cache_dir}")
        
    except Exception as e:
        print(f"  ✗ Failed to download T5: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Download pretrained models")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="models/pretrained",
        help="Directory to cache models",
    )
    parser.add_argument(
        "--t5_model",
        type=str,
        default="t5-small",
        help="T5 model to download (t5-small, t5-base)",
    )
    args = parser.parse_args()
    
    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    print("="*50)
    print("Downloading Pretrained Models")
    print("="*50)
    print(f"Cache directory: {cache_dir}\n")
    
    # Download CLIP
    clip_success = download_clip(cache_dir)
    
    # Download T5
    t5_success = download_t5(cache_dir, args.t5_model)
    
    print("\n" + "="*50)
    if clip_success and t5_success:
        print("All models downloaded successfully!")
        print("You can now run experiments offline.")
    else:
        print("Some downloads failed. Check the errors above.")
    print("="*50)


if __name__ == "__main__":
    main()
