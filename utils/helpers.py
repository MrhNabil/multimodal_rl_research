"""
Helper Utilities

General utility functions for reproducibility,
configuration management, and common operations.
"""

import os
import random
from typing import Dict, Any, Optional
import yaml

import torch
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration with overrides
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device() -> str:
    """
    Get the available device.
    
    Auto-detects GPU if available.
    
    Returns:
        Device string
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        The same path
    """
    os.makedirs(path, exist_ok=True)
    return path


class EarlyStopping:
    """Early stopping helper for training."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of evaluations to wait
            min_delta: Minimum change to qualify as improvement
            mode: "max" for accuracy, "min" for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.counter = 0
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


def main():
    """Test helper utilities."""
    print("Testing helper utilities...")
    
    # Test seed setting
    set_seed(42)
    print(f"Random value 1: {random.random()}")
    set_seed(42)
    print(f"Random value 2: {random.random()}")
    
    # Test config merge
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 10}, "e": 5}
    merged = merge_configs(base, override)
    print(f"Merged config: {merged}")
    
    # Test device
    print(f"Device: {get_device()}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3)
    values = [0.5, 0.6, 0.55, 0.55, 0.55, 0.55]
    for v in values:
        stop = early_stop(v)
        print(f"Value: {v}, Stop: {stop}")


if __name__ == "__main__":
    main()
