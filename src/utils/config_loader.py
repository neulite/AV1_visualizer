"""Configuration file loader for AV1 Visualizer

This module provides functions to load YAML configuration files.
"""

import yaml
from pathlib import Path


def load_config(config_path='config/paths.yaml'):
    """Load YAML configuration file

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file is not found
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please copy config/paths_template.yaml to create one."
        )

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
