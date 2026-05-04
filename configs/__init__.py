"""VeraRAG configuration management."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

CONFIG_DIR = Path(__file__).parent


def load_config(config_name: str = "model.yaml") -> Dict[str, Any]:
    """
    Load a configuration file.

    Args:
        config_name: Name of the config file (e.g., "model.yaml", "hotpotqa.yaml")

    Returns:
        Configuration dictionary
    """
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

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


def get_model_config() -> Dict[str, Any]:
    """Get the model configuration."""
    return load_config("model.yaml")


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset-specific configuration.

    Args:
        dataset_name: Name of the dataset (hotpotqa, fever, ckt_conflict)

    Returns:
        Dataset configuration dictionary
    """
    base_config = get_model_config()
    dataset_config = load_config(f"{dataset_name}.yaml")
    return merge_configs(base_config, dataset_config)
