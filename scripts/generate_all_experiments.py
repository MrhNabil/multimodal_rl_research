#!/usr/bin/env python
"""
Generate All 61 Experiment Configuration Files

Creates experiment configs for the complete hyperparameter sweep:
- exp_001-003: Baselines (frozen, supervised, RL)
- exp_004-013: Learning rates
- exp_014-023: Reward functions
- exp_024-033: Question types
- exp_034-043: Baseline types and decay values
- exp_044-051: Temperature values
- exp_052-056: Entropy coefficients
- exp_057-061: Batch sizes
"""

import os
import yaml

EXPERIMENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "experiments"
)


def create_config(exp_id: int, name: str, overrides: dict) -> dict:
    """Create an experiment config with the given overrides."""
    base = {
        "experiment": {
            "id": exp_id,
            "name": name,
        },
        "seed": 42,
        "training": {
            "method": "rl",
            "batch_size": 32,
            "max_steps": 5000,
            "eval_every": 100,
            "save_every": 500,
            "optimizer": {
                "learning_rate": 0.0001,
            },
        },
        "reinforce": {
            "reward_type": "exact_match",
            "baseline_type": "moving_avg",
            "baseline_decay": 0.99,
            "temperature": 1.0,
            "entropy_coef": 0.01,
        },
    }
    
    # Apply overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base:
            base[key].update(value)
        else:
            base[key] = value
    
    return base


def save_config(config: dict, exp_id: int, suffix: str):
    """Save config to YAML file."""
    filename = f"exp_{exp_id:03d}_{suffix}.yaml"
    filepath = os.path.join(EXPERIMENTS_DIR, filename)
    
    # Add comment header
    comment = f"# Experiment {exp_id:03d}: {suffix.replace('_', ' ').title()}\n"
    
    with open(filepath, "w") as f:
        f.write(comment)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"  Created: {filename}")


def generate_all_experiments():
    """Generate all 61 experiment configuration files."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    
    print("Generating experiment configurations...")
    print("=" * 50)
    
    # =========================================================================
    # Experiments 001-003: Baselines (already exist, but regenerate for consistency)
    # =========================================================================
    print("\n[001-003] Baseline Methods")
    
    # exp_001: Frozen baseline
    config = create_config(1, "frozen_baseline", {})
    config["training"]["method"] = "frozen"
    save_config(config, 1, "frozen")
    
    # exp_002: Supervised baseline
    config = create_config(2, "supervised_baseline", {})
    config["training"]["method"] = "supervised"
    save_config(config, 2, "supervised")
    
    # exp_003: RL baseline
    config = create_config(3, "rl_baseline", {})
    save_config(config, 3, "rl_baseline")
    
    # =========================================================================
    # Experiments 004-013: Learning Rates
    # =========================================================================
    print("\n[004-013] Learning Rates")
    
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    for i, lr in enumerate(learning_rates, start=4):
        config = create_config(i, f"lr_{lr}", {
            "training": {"optimizer": {"learning_rate": lr}}
        })
        lr_str = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
        save_config(config, i, f"lr_{lr_str}")
    
    # =========================================================================
    # Experiments 014-023: Reward Functions
    # =========================================================================
    print("\n[014-023] Reward Functions")
    
    reward_configs = [
        ("exact_match", {}),
        ("partial_match", {}),
        ("length_penalty", {"length_penalty": 0.1}),
        ("combined", {"weights": [0.8, 0.2]}),
        ("progressive", {"transition_steps": 1000}),
        ("exact_strict", {"case_sensitive": True}),
        ("partial_strict", {"case_sensitive": True}),
        ("combined_heavy_length", {"weights": [0.6, 0.4]}),
        ("progressive_fast", {"transition_steps": 500}),
        ("progressive_slow", {"transition_steps": 2000}),
    ]
    
    for i, (reward_name, extra) in enumerate(reward_configs, start=14):
        config = create_config(i, f"reward_{reward_name}", {
            "reinforce": {"reward_type": reward_name.split("_")[0], **extra}
        })
        save_config(config, i, f"reward_{reward_name}")
    
    # =========================================================================
    # Experiments 024-033: Question Types
    # =========================================================================
    print("\n[024-033] Question Types")
    
    question_configs = [
        ("color_only", ["color"]),
        ("shape_only", ["shape"]),
        ("count_only", ["count"]),
        ("spatial_only", ["spatial"]),
        ("color_shape", ["color", "shape"]),
        ("color_count", ["color", "count"]),
        ("shape_count", ["shape", "count"]),
        ("color_spatial", ["color", "spatial"]),
        ("no_spatial", ["color", "shape", "count"]),
        ("all_types", ["color", "shape", "count", "spatial"]),
    ]
    
    for i, (name, qtypes) in enumerate(question_configs, start=24):
        config = create_config(i, f"qtype_{name}", {
            "data": {"question_types": qtypes}
        })
        save_config(config, i, f"qtype_{name}")
    
    # =========================================================================
    # Experiments 034-043: Baseline Types and Decay
    # =========================================================================
    print("\n[034-043] Baseline Configurations")
    
    baseline_configs = [
        ("none", "none", 0.0),
        ("moving_avg_0.9", "moving_avg", 0.9),
        ("moving_avg_0.95", "moving_avg", 0.95),
        ("moving_avg_0.99", "moving_avg", 0.99),
        ("moving_avg_0.999", "moving_avg", 0.999),
        ("learned", "learned", 0.0),
        ("moving_avg_0.8", "moving_avg", 0.8),
        ("moving_avg_0.7", "moving_avg", 0.7),
        ("moving_avg_0.5", "moving_avg", 0.5),
        ("moving_avg_0.0", "moving_avg", 0.0),
    ]
    
    for i, (name, btype, decay) in enumerate(baseline_configs, start=34):
        config = create_config(i, f"baseline_{name}", {
            "reinforce": {"baseline_type": btype, "baseline_decay": decay}
        })
        save_config(config, i, f"baseline_{name}")
    
    # =========================================================================
    # Experiments 044-051: Temperature
    # =========================================================================
    print("\n[044-051] Temperature Values")
    
    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    for i, temp in enumerate(temperatures, start=44):
        config = create_config(i, f"temp_{temp}", {
            "reinforce": {"temperature": temp}
        })
        temp_str = str(temp).replace(".", "_")
        save_config(config, i, f"temp_{temp_str}")
    
    # =========================================================================
    # Experiments 052-056: Entropy Coefficient
    # =========================================================================
    print("\n[052-056] Entropy Coefficients")
    
    entropy_coefs = [0.0, 0.001, 0.01, 0.05, 0.1]
    for i, ent in enumerate(entropy_coefs, start=52):
        config = create_config(i, f"entropy_{ent}", {
            "reinforce": {"entropy_coef": ent}
        })
        ent_str = str(ent).replace(".", "_")
        save_config(config, i, f"entropy_{ent_str}")
    
    # =========================================================================
    # Experiments 057-061: Batch Size
    # =========================================================================
    print("\n[057-061] Batch Sizes")
    
    batch_sizes = [8, 16, 32, 64, 128]
    for i, bs in enumerate(batch_sizes, start=57):
        config = create_config(i, f"batch_{bs}", {
            "training": {"batch_size": bs}
        })
        save_config(config, i, f"batch_{bs}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("Generation complete!")
    
    # Count files
    files = [f for f in os.listdir(EXPERIMENTS_DIR) if f.endswith(".yaml")]
    print(f"Total experiment configs: {len(files)}")
    print(f"Location: {EXPERIMENTS_DIR}")


if __name__ == "__main__":
    generate_all_experiments()
