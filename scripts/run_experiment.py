#!/usr/bin/env python
"""
Run Single Experiment

Runs a single experiment with the specified configuration.
Supports frozen baseline, supervised training, and RL training.
"""

import os
import argparse
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils.helpers import set_seed, load_config, merge_configs, ensure_dir
from utils.logging import ExperimentLogger
from data.dataset import MultimodalDataset, create_dataloader
from models.multimodal import create_multimodal_vqa
from training.reinforce import REINFORCETrainer, TrainingConfig
from training.supervised import SupervisedTrainer, SupervisedConfig
from training.rewards import create_reward_function
from evaluation.evaluator import Evaluator


def run_frozen_baseline(model, val_dataloader, test_dataloader, config, logger):
    """Run frozen baseline (no training)."""
    print("Running frozen baseline (no training)...")
    
    evaluator = Evaluator(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=logger.get_run_dir(),
    )
    
    # Evaluate on validation set
    val_results = evaluator.evaluate(val_dataloader, split_name="val")
    
    # Evaluate on test set
    test_results = evaluator.evaluate(test_dataloader, split_name="test")
    
    # Log final results
    logger.log_final_results({
        "method": "frozen",
        "accuracy": test_results.accuracy,
        "val_accuracy": val_results.accuracy,
        **config,
    })
    
    return test_results


def run_supervised_training(model, train_dataloader, val_dataloader, test_dataloader, config, logger):
    """Run supervised training baseline."""
    print("Running supervised training...")
    
    trainer_config = SupervisedConfig(
        batch_size=config.get("batch_size", 32),
        max_steps=config.get("max_steps", 5000),
        learning_rate=config.get("learning_rate", 1e-4),
        eval_every=config.get("eval_every", 100),
        save_every=config.get("save_every", 500),
        output_dir=logger.get_run_dir(),
        experiment_name="training",
    )
    
    trainer = SupervisedTrainer(model, trainer_config)
    trainer.train(train_dataloader, val_dataloader)
    
    # Final evaluation
    evaluator = Evaluator(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=logger.get_run_dir(),
    )
    
    test_results = evaluator.evaluate(test_dataloader, split_name="test")
    
    logger.log_final_results({
        "method": "supervised",
        "accuracy": test_results.accuracy,
        "best_val_accuracy": trainer.best_accuracy,
        **config,
    })
    
    return test_results


def run_rl_training(model, train_dataloader, val_dataloader, test_dataloader, config, logger):
    """Run REINFORCE training."""
    print("Running REINFORCE training...")
    
    # Create reward function
    reward_fn = create_reward_function(
        reward_type=config.get("reward_type", "exact_match"),
    )
    
    trainer_config = TrainingConfig(
        batch_size=config.get("batch_size", 32),
        max_steps=config.get("max_steps", 5000),
        learning_rate=config.get("learning_rate", 1e-4),
        eval_every=config.get("eval_every", 100),
        save_every=config.get("save_every", 500),
        reward_type=config.get("reward_type", "exact_match"),
        baseline_type=config.get("baseline_type", "moving_avg"),
        temperature=config.get("temperature", 1.0),
        entropy_coef=config.get("entropy_coef", 0.01),
        output_dir=logger.get_run_dir(),
        experiment_name="training",
    )
    
    trainer = REINFORCETrainer(
        model=model,
        config=trainer_config,
        reward_fn=reward_fn,
    )
    
    trainer.train(train_dataloader, val_dataloader)
    
    # Final evaluation
    evaluator = Evaluator(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=logger.get_run_dir(),
    )
    
    test_results = evaluator.evaluate(test_dataloader, split_name="test")
    
    logger.log_final_results({
        "method": "rl",
        "accuracy": test_results.accuracy,
        "best_val_accuracy": trainer.best_accuracy,
        **config,
    })
    
    return test_results


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/generated",
        help="Directory containing generated data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--use_dummy",
        action="store_true",
        help="Use dummy models for testing",
    )
    args = parser.parse_args()
    
    # Load configuration
    base_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "base_config.yaml",
    )
    
    base_config = load_config(base_config_path)
    exp_config = load_config(args.config)
    config = merge_configs(base_config, exp_config)
    
    # Override from command line
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    if args.seed:
        config["seed"] = args.seed
    
    # Generate experiment name from config path
    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    
    print("="*60)
    print(f"Running Experiment: {exp_name}")
    print("="*60)
    print(f"Method: {config.get('training', {}).get('method', 'rl')}")
    print(f"Seed: {config.get('seed', 42)}")
    print(f"Max steps: {config.get('training', {}).get('max_steps', 5000)}")
    print("="*60 + "\n")
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Create logger
    logger = ExperimentLogger(
        log_dir=args.output_dir,
        experiment_name=exp_name,
        config=config,
    )
    
    # Load datasets
    print("Loading datasets...")
    
    # Get CLIP preprocessing
    if not args.use_dummy:
        from models.vision import create_vision_encoder
        temp_encoder = create_vision_encoder(use_dummy=False)
        transform = temp_encoder.get_preprocess()
    else:
        transform = None
    
    train_dataset = MultimodalDataset(
        data_dir=args.data_dir,
        split="train",
        transform=transform,
    )
    
    val_dataset = MultimodalDataset(
        data_dir=args.data_dir,
        split="val",
        transform=transform,
    )
    
    test_dataset = MultimodalDataset(
        data_dir=args.data_dir,
        split="test",
        transform=transform,
    )
    
    batch_size = config.get("training", {}).get("batch_size", 32)
    
    train_dataloader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = create_multimodal_vqa(
        use_dummy=args.use_dummy,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    param_counts = model.count_parameters()
    print(f"Model parameters: {param_counts}")
    
    # Run experiment
    method = config.get("training", {}).get("method", "rl")
    training_config = {
        "batch_size": batch_size,
        "max_steps": config.get("training", {}).get("max_steps", 5000),
        "learning_rate": config.get("training", {}).get("optimizer", {}).get("learning_rate", 1e-4),
        "eval_every": config.get("training", {}).get("eval_every", 100),
        "save_every": config.get("training", {}).get("save_every", 500),
        "reward_type": config.get("reinforce", {}).get("reward_type", "exact_match"),
        "baseline_type": config.get("reinforce", {}).get("baseline_type", "moving_avg"),
        "temperature": config.get("reinforce", {}).get("temperature", 1.0),
        "entropy_coef": config.get("reinforce", {}).get("entropy_coef", 0.01),
        "seed": config.get("seed", 42),
    }
    
    start_time = time.time()
    
    if method == "frozen":
        results = run_frozen_baseline(
            model, val_dataloader, test_dataloader, training_config, logger
        )
    elif method == "supervised":
        results = run_supervised_training(
            model, train_dataloader, val_dataloader, test_dataloader, training_config, logger
        )
    else:  # rl
        results = run_rl_training(
            model, train_dataloader, val_dataloader, test_dataloader, training_config, logger
        )
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
    print(f"Final accuracy: {results.accuracy:.4f}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Results saved to: {logger.get_run_dir()}")
    print("="*60)


if __name__ == "__main__":
    main()
