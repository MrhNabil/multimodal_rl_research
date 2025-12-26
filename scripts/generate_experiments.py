#!/usr/bin/env python
"""
Generate Experiment Configuration Files

Creates YAML configuration files for all 61 experiments.
Run this script to populate the config/experiments directory.
"""

import os
import yaml

# Base output directory
CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "experiments"
)


def create_config(
    exp_id: int,
    name: str,
    method: str,
    seed: int = 42,
    **overrides
):
    """Create an experiment configuration."""
    config = {
        "experiment": {
            "id": exp_id,
            "name": name,
        },
        "seed": seed,
        "training": {
            "method": method,
            "batch_size": overrides.get("batch_size", 32),
            "max_steps": overrides.get("max_steps", 5000),
            "eval_every": 100,
            "save_every": 500,
            "optimizer": {
                "learning_rate": overrides.get("learning_rate", 1e-4),
            },
        },
    }
    
    # Add REINFORCE-specific settings
    if method == "rl":
        config["reinforce"] = {
            "reward_type": overrides.get("reward_type", "exact_match"),
            "baseline_type": overrides.get("baseline_type", "moving_avg"),
            "temperature": overrides.get("temperature", 1.0),
            "entropy_coef": overrides.get("entropy_coef", 0.01),
        }
    
    # Add question type filter if specified
    if "question_types" in overrides:
        config["data"] = {
            "question_types": overrides["question_types"],
        }
    
    return config


def save_config(config, exp_id, name):
    """Save configuration to YAML file."""
    filename = f"exp_{exp_id:03d}_{name}.yaml"
    filepath = os.path.join(CONFIG_DIR, filename)
    
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return filepath


def generate_all_configs():
    """Generate all 61 experiment configurations."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    experiments = []
    exp_id = 1
    
    # =========================================
    # Experiments 001-003: Baseline Methods
    # =========================================
    
    # 001: Frozen baseline
    config = create_config(exp_id, "frozen", method="frozen")
    save_config(config, exp_id, "frozen")
    experiments.append((exp_id, "frozen", "Frozen baseline (no training)"))
    exp_id += 1
    
    # 002: Supervised baseline
    config = create_config(exp_id, "supervised", method="supervised")
    save_config(config, exp_id, "supervised")
    experiments.append((exp_id, "supervised", "Supervised cross-entropy baseline"))
    exp_id += 1
    
    # 003: RL baseline
    config = create_config(exp_id, "rl_baseline", method="rl")
    save_config(config, exp_id, "rl_baseline")
    experiments.append((exp_id, "rl_baseline", "REINFORCE baseline"))
    exp_id += 1
    
    # =========================================
    # Experiments 004-013: Learning Rate Sweep
    # =========================================
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    seeds = [42, 123]
    
    for lr in learning_rates:
        for seed in seeds:
            name = f"lr_{lr:.0e}_seed_{seed}".replace("-", "m").replace("+", "p")
            config = create_config(
                exp_id, name, method="rl",
                learning_rate=lr, seed=seed
            )
            save_config(config, exp_id, name)
            experiments.append((exp_id, name, f"LR={lr}, seed={seed}"))
            exp_id += 1
    
    # =========================================
    # Experiments 014-023: Reward Function Variations
    # =========================================
    reward_types = ["exact_match", "partial_match", "combined", "progressive"]
    seeds = [42, 123]
    
    for reward in reward_types:
        for seed in seeds:
            name = f"reward_{reward}_seed_{seed}"
            config = create_config(
                exp_id, name, method="rl",
                reward_type=reward, seed=seed
            )
            save_config(config, exp_id, name)
            experiments.append((exp_id, name, f"Reward={reward}, seed={seed}"))
            exp_id += 1
    
    # Add two more for padding to 10
    for extra in range(2):
        name = f"reward_exact_lr_high_seed_{42 + extra}"
        config = create_config(
            exp_id, name, method="rl",
            reward_type="exact_match", learning_rate=5e-4, seed=42 + extra
        )
        save_config(config, exp_id, name)
        experiments.append((exp_id, name, f"Exact match + high LR, seed={42 + extra}"))
        exp_id += 1
    
    # =========================================
    # Experiments 024-033: Question Type Ablations
    # =========================================
    question_types = ["color", "shape", "count", "spatial"]
    methods = ["supervised", "rl"]
    
    for q_type in question_types:
        for method in methods:
            name = f"qtype_{q_type}_{method}"
            config = create_config(
                exp_id, name, method=method,
                question_types=[q_type]
            )
            save_config(config, exp_id, name)
            experiments.append((exp_id, name, f"Question type={q_type}, method={method}"))
            exp_id += 1
    
    # All question types comparison
    for method in methods:
        name = f"qtype_all_{method}"
        config = create_config(
            exp_id, name, method=method,
            question_types=["color", "shape", "count", "spatial"]
        )
        save_config(config, exp_id, name)
        experiments.append((exp_id, name, f"All question types, method={method}"))
        exp_id += 1
    
    # =========================================
    # Experiments 034-043: Baseline Variations
    # =========================================
    baseline_types = ["none", "moving_avg", "learned"]
    seeds = [42, 123, 456]
    
    for baseline in baseline_types:
        for seed in seeds[:2]:  # 2 seeds per baseline
            name = f"baseline_{baseline}_seed_{seed}"
            config = create_config(
                exp_id, name, method="rl",
                baseline_type=baseline, seed=seed
            )
            save_config(config, exp_id, name)
            experiments.append((exp_id, name, f"Baseline={baseline}, seed={seed}"))
            exp_id += 1
    
    # Different decay rates for moving average
    for decay in [0.9, 0.95, 0.99, 0.999]:
        name = f"baseline_mavg_decay_{decay}"
        config = create_config(
            exp_id, name, method="rl",
            baseline_type="moving_avg"
        )
        config["reinforce"]["baseline_decay"] = decay
        save_config(config, exp_id, name)
        experiments.append((exp_id, name, f"Moving avg decay={decay}"))
        exp_id += 1
    
    # =========================================
    # Experiments 044-053: Temperature Variations  
    # =========================================
    temperatures = [0.5, 0.7, 1.0, 1.5]
    seeds = [42, 123]
    
    for temp in temperatures:
        for seed in seeds:
            name = f"temp_{temp}_seed_{seed}".replace(".", "p")
            config = create_config(
                exp_id, name, method="rl",
                temperature=temp, seed=seed
            )
            save_config(config, exp_id, name)
            experiments.append((exp_id, name, f"Temperature={temp}, seed={seed}"))
            exp_id += 1
    
    # =========================================
    # Experiments 054-058: Entropy Coefficient
    # =========================================
    entropy_coefs = [0.0, 0.01, 0.05, 0.1, 0.2]
    
    for ent in entropy_coefs:
        name = f"entropy_{ent}".replace(".", "p")
        config = create_config(
            exp_id, name, method="rl",
            entropy_coef=ent
        )
        save_config(config, exp_id, name)
        experiments.append((exp_id, name, f"Entropy coef={ent}"))
        exp_id += 1
    
    # =========================================
    # Experiments 059-063: Batch Size Variations
    # =========================================
    batch_sizes = [8, 16, 32, 64, 128]
    
    for bs in batch_sizes:
        name = f"batch_{bs}"
        config = create_config(
            exp_id, name, method="rl",
            batch_size=bs
        )
        save_config(config, exp_id, name)
        experiments.append((exp_id, name, f"Batch size={bs}"))
        exp_id += 1
    
    return experiments


def main():
    print("="*60)
    print("Generating Experiment Configurations")
    print("="*60)
    
    experiments = generate_all_configs()
    
    print(f"\nGenerated {len(experiments)} experiment configurations:")
    print(f"Output directory: {CONFIG_DIR}\n")
    
    # Print summary by category
    categories = [
        ("Baseline Methods", 1, 3),
        ("Learning Rate Sweep", 4, 13),
        ("Reward Functions", 14, 23),
        ("Question Types", 24, 33),
        ("Baseline Variations", 34, 43),
        ("Temperature", 44, 51),
        ("Entropy Coefficient", 52, 56),
        ("Batch Size", 57, 61),
    ]
    
    for cat_name, start, end in categories:
        count = end - start + 1
        print(f"  {cat_name}: experiments {start:03d}-{end:03d} ({count} experiments)")
    
    print(f"\nTotal: {len(experiments)} experiments")
    print("="*60)


if __name__ == "__main__":
    main()
