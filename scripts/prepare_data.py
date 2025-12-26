#!/usr/bin/env python
"""
Prepare Dataset

Generates the synthetic CLEVR-like dataset for VQA experiments.
Creates train, validation, and test splits with images and questions.
"""

import os
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_clevr import SyntheticCLEVRGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic VQA dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/generated",
        help="Directory to save generated data",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=5000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=1000,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=1000,
        help="Number of test samples",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Size of generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--question_types",
        type=str,
        nargs="+",
        default=["color", "shape", "count", "spatial"],
        help="Types of questions to generate",
    )
    parser.add_argument(
        "--holdout",
        type=str,
        nargs="*",
        default=None,
        help="Holdout color-shape pairs (e.g., 'red-cube' 'blue-sphere')",
    )
    args = parser.parse_args()
    
    print("="*50)
    print("Synthetic Dataset Generator")
    print("="*50)
    print(f"Output directory: {args.output_dir}")
    print(f"Training samples: {args.num_train}")
    print(f"Validation samples: {args.num_val}")
    print(f"Test samples: {args.num_test}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Seed: {args.seed}")
    print(f"Question types: {args.question_types}")
    
    # Parse holdout combinations
    holdout_combinations = None
    if args.holdout:
        holdout_combinations = []
        for combo in args.holdout:
            parts = combo.split("-")
            if len(parts) == 2:
                holdout_combinations.append(tuple(parts))
        print(f"Holdout combinations: {holdout_combinations}")
    
    print("="*50 + "\n")
    
    # Create generator
    generator = SyntheticCLEVRGenerator(
        output_dir=args.output_dir,
        image_size=args.image_size,
        seed=args.seed,
    )
    
    # Generate dataset
    datasets = generator.generate_dataset(
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        question_types=args.question_types,
        holdout_combinations=holdout_combinations,
    )
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset Generation Complete!")
    print("="*50)
    
    for split, samples in datasets.items():
        print(f"\n{split.upper()} split:")
        print(f"  Total samples: {len(samples)}")
        
        # Count question types
        type_counts = {}
        for sample in samples:
            q_type = sample.question_type
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        
        print(f"  Question type distribution:")
        for q_type, count in sorted(type_counts.items()):
            print(f"    {q_type}: {count}")
    
    print(f"\nData saved to: {args.output_dir}")
    print("\nExample samples:")
    for i, sample in enumerate(datasets["train"][:3]):
        print(f"\n  Sample {i+1}:")
        print(f"    Question: {sample.question}")
        print(f"    Answer: {sample.answer}")
        print(f"    Type: {sample.question_type}")


if __name__ == "__main__":
    main()
