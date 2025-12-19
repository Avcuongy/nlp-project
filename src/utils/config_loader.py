from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def _import_yaml():
    """Import PyYAML library, raising an informative error if not installed."""
    try:
        import yaml  # type: ignore

        return yaml
    except Exception as exc:
        raise ImportError(
            "PyYAML is required to load YAML config files. Please 'pip install pyyaml'."
        ) from exc


def load_yaml_file(path: str | os.PathLike) -> Dict[str, Any]:
    """
    Load a YAML file into a Python dictionary using safe loader.

    Args:
        path (str | os.PathLike): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML content as a dictionary.
    """
    yaml = _import_yaml()
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def expand_env_vars(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Expand environment variables in all string values of a nested config dict.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Configuration with environment variables expanded.
    """

    def _expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, list):
            return [_expand(v) for v in value]
        if isinstance(value, dict):
            return {k: _expand(v) for k, v in value.items()}
        return value

    return _expand(cfg)


def load_config(config_path: str | os.PathLike) -> Dict[str, Any]:
    """Load a configuration YAML file and expand environment variables.

    Args:
        config_path (str | os.PathLike): Path to YAML config file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    cfg = load_yaml_file(config_path)
    cfg = expand_env_vars(cfg)
    return cfg
