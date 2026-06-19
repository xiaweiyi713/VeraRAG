"""VeraRAG configuration management."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent


def load_config(config_name: str = "model.yaml") -> dict[str, Any]:
    """
    Load a configuration file.

    Args:
        config_name: Name of the config file (e.g., "model.yaml", "hotpotqa.yaml")

    Returns:
        Configuration dictionary
    """
    config_path = _resolve_config_path(config_name)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return config


def merge_configs(
    base_config: dict[str, Any],
    override_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    result = deepcopy(base_config)
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_model_config() -> dict[str, Any]:
    """Get the model configuration."""
    return load_config("model.yaml")


def get_dataset_config(dataset_name: str) -> dict[str, Any]:
    """
    Get dataset-specific configuration.

    Args:
        dataset_name: Name of the dataset (hotpotqa, fever, ckt_conflict)

    Returns:
        Dataset configuration dictionary
    """
    base_config = get_model_config()
    config_name = dataset_name if dataset_name.endswith(".yaml") else f"{dataset_name}.yaml"
    dataset_config = load_config(config_name)
    return merge_configs(base_config, dataset_config)


def _resolve_config_path(config_name: str) -> Path:
    requested = Path(config_name)
    if requested.is_absolute() or requested.suffix != ".yaml":
        raise ValueError("Config name must be a relative .yaml file inside configs/")
    config_root = CONFIG_DIR.resolve()
    config_path = (config_root / requested).resolve()
    try:
        config_path.relative_to(config_root)
    except ValueError as exc:
        raise ValueError("Config name must stay inside configs/") from exc
    return config_path
