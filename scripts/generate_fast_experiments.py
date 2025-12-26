#!/usr/bin/env python
"""
Generate Fast CPU Experiments (50-100 steps each)

Creates 61 experiment configs optimized for:
- CPU-only execution
- Fast completion (seconds per experiment)
- Comprehensive hyperparameter coverage
"""

import os
import yaml

EXPERIMENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "experiments"
)

# CONFIG: Quick iteration settings (1000 steps)
FAST_SETTINGS = {
    "max_steps": 1000,    # 1000 steps for better accuracy (~3-4 min per experiment)
    "batch_size": 64,     # Large batches for stability
    "eval_every": 200,    # Evaluate 5 times
    "save_every": 1000,   # Save at end
    "log_every": 100,
    "learning_rate": 2e-4,  # Best LR from previous experiments
}


def create_config(exp_id: int, name: str, overrides: dict) -> dict:
    """Create an experiment config with fast settings."""
    base = {
        "experiment": {
            "id": exp_id,
            "name": name,
        },
        "seed": 42,
        "training": {
            "method": "rl",
            "batch_size": FAST_SETTINGS["batch_size"],
            "max_steps": FAST_SETTINGS["max_steps"],
            "eval_every": FAST_SETTINGS["eval_every"],
            "save_every": FAST_SETTINGS["save_every"],
            "optimizer": {
                "learning_rate": FAST_SETTINGS.get("learning_rate", 1e-3),
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
    
    comment = f"# Experiment {exp_id:03d}: {suffix.replace('_', ' ').title()} [FAST MODE: {FAST_SETTINGS['max_steps']} steps]\n"
    
    with open(filepath, "w") as f:
        f.write(comment)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return filename


def generate_all_experiments():
    """Generate all 61 fast experiment configurations."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    
    # Clear existing experiments
    for f in os.listdir(EXPERIMENTS_DIR):
        if f.startswith("exp_") and f.endswith(".yaml"):
            os.remove(os.path.join(EXPERIMENTS_DIR, f))
    
    created = []
    
    print("=" * 60)
    print("FAST EXPERIMENT GENERATOR")
    print(f"Settings: {FAST_SETTINGS['max_steps']} steps, batch_size={FAST_SETTINGS['batch_size']}")
    print("=" * 60)
    
    # =========================================================================
    # 001-003: Baseline Methods
    # =========================================================================
    print("\n[001-003] Baseline Methods")
    
    config = create_config(1, "frozen_baseline", {})
    config["training"]["method"] = "frozen"
    created.append(save_config(config, 1, "frozen"))
    
    config = create_config(2, "supervised_baseline", {})
    config["training"]["method"] = "supervised"
    created.append(save_config(config, 2, "supervised"))
    
    config = create_config(3, "rl_baseline", {})
    created.append(save_config(config, 3, "rl_baseline"))
    
    # =========================================================================
    # 004-013: Learning Rates (10 experiments)
    # =========================================================================
    print("[004-013] Learning Rates")
    
    lrs = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    for i, lr in enumerate(lrs, start=4):
        config = create_config(i, f"lr_{lr}", {
            "training": {"optimizer": {"learning_rate": lr}}
        })
        lr_str = f"{lr:.0e}".replace("-0", "-").replace("+0", "")
        created.append(save_config(config, i, f"lr_{lr_str}"))
    
    # =========================================================================
    # 014-023: Reward Functions (10 experiments)
    # =========================================================================
    print("[014-023] Reward Functions")
    
    rewards = [
        ("exact_match", {}),
        ("partial_match", {}),
        ("length_penalty", {}),
        ("combined", {}),
        ("progressive", {}),
        ("exact_strict", {}),
        ("partial_strict", {}),
        ("combined_v2", {}),
        ("progressive_fast", {}),
        ("progressive_slow", {}),
    ]
    for i, (name, _) in enumerate(rewards, start=14):
        config = create_config(i, f"reward_{name}", {
            "reinforce": {"reward_type": name.split("_")[0]}
        })
        created.append(save_config(config, i, f"reward_{name}"))
    
    # =========================================================================
    # 024-033: Question Types (10 experiments)
    # =========================================================================
    print("[024-033] Question Types")
    
    qtypes = [
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
    for i, (name, types) in enumerate(qtypes, start=24):
        config = create_config(i, f"qtype_{name}", {
            "data": {"question_types": types}
        })
        created.append(save_config(config, i, f"qtype_{name}"))
    
    # =========================================================================
    # 034-043: Baseline Configurations (10 experiments)
    # =========================================================================
    print("[034-043] Baseline Configurations")
    
    baselines = [
        ("none", "none", 0.0),
        ("mavg_0.9", "moving_avg", 0.9),
        ("mavg_0.95", "moving_avg", 0.95),
        ("mavg_0.99", "moving_avg", 0.99),
        ("mavg_0.999", "moving_avg", 0.999),
        ("learned", "learned", 0.0),
        ("mavg_0.8", "moving_avg", 0.8),
        ("mavg_0.7", "moving_avg", 0.7),
        ("mavg_0.5", "moving_avg", 0.5),
        ("mavg_0.0", "moving_avg", 0.0),
    ]
    for i, (name, btype, decay) in enumerate(baselines, start=34):
        config = create_config(i, f"baseline_{name}", {
            "reinforce": {"baseline_type": btype, "baseline_decay": decay}
        })
        created.append(save_config(config, i, f"baseline_{name}"))
    
    # =========================================================================
    # 044-051: Temperature (8 experiments)
    # =========================================================================
    print("[044-051] Temperature Values")
    
    temps = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    for i, temp in enumerate(temps, start=44):
        config = create_config(i, f"temp_{temp}", {
            "reinforce": {"temperature": temp}
        })
        created.append(save_config(config, i, f"temp_{str(temp).replace('.', '_')}"))
    
    # =========================================================================
    # 052-056: Entropy Coefficient (5 experiments)
    # =========================================================================
    print("[052-056] Entropy Coefficients")
    
    entropies = [0.0, 0.001, 0.01, 0.05, 0.1]
    for i, ent in enumerate(entropies, start=52):
        config = create_config(i, f"entropy_{ent}", {
            "reinforce": {"entropy_coef": ent}
        })
        created.append(save_config(config, i, f"entropy_{str(ent).replace('.', '_')}"))
    
    # =========================================================================
    # 057-061: Batch Sizes (5 experiments)
    # =========================================================================
    print("[057-061] Batch Sizes")
    
    batch_sizes = [4, 8, 16, 32, 64]
    for i, bs in enumerate(batch_sizes, start=57):
        config = create_config(i, f"batch_{bs}", {
            "training": {"batch_size": bs}
        })
        created.append(save_config(config, i, f"batch_{bs}"))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"[OK] Generated {len(created)} experiment configs")
    print(f"Location: {EXPERIMENTS_DIR}")
    print(f"Each experiment: ~{FAST_SETTINGS['max_steps']} steps")
    print("=" * 60)
    
    return created


if __name__ == "__main__":
    generate_all_experiments()
