from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping

from .common import ensure_dir


def _import_yaml():
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
	"""
	yaml = _import_yaml()
	with Path(path).open("r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def save_yaml_file(path: str | os.PathLike, data: Mapping[str, Any]) -> None:
	"""Save a mapping to a YAML file (UTF-8)."""
	yaml = _import_yaml()
	p = Path(path)
	ensure_dir(p.parent)
	with p.open("w", encoding="utf-8") as f:
		yaml.safe_dump(dict(data), f, allow_unicode=True, sort_keys=False)


def deep_update(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
	"""
	Recursively update a nested dictionary with another mapping.
	Returns a new dictionary; does not mutate inputs.
	"""
	result: Dict[str, Any] = dict(base)
	for k, v in overrides.items():
		if isinstance(v, Mapping) and isinstance(result.get(k), Mapping):
			result[k] = deep_update(result[k], v)  # type: ignore[index]
		else:
			result[k] = v
	return result


def expand_env_vars(cfg: Dict[str, Any]) -> Dict[str, Any]:
	"""Expand environment variables in all string values of a nested config dict."""
	def _expand(value: Any) -> Any:
		if isinstance(value, str):
			return os.path.expandvars(value)
		if isinstance(value, list):
			return [_expand(v) for v in value]
		if isinstance(value, dict):
			return {k: _expand(v) for k, v in value.items()}
		return value

	return _expand(cfg)


def load_project_config(config_path: str | os.PathLike) -> Dict[str, Any]:
	"""
	Load the main project config (config/config.yaml), expand env vars,
	and normalize common path fields to POSIX-style strings.
	"""
	cfg = load_yaml_file(config_path)
	cfg = expand_env_vars(cfg)

	# Normalize path fields for downstream code
	for key in ("paths", "logging"):
		if key in cfg and isinstance(cfg[key], dict):
			for k, v in list(cfg[key].items()):
				if isinstance(v, str):
					cfg[key][k] = str(Path(v))
	return cfg


def load_model_config(project_cfg: Mapping[str, Any]) -> Dict[str, Any]:
	"""
	Resolve and load the model-specific YAML based on experiment.model_family.
	Merge project-level hints (if any) onto the model config.
	"""
	exp = project_cfg.get("experiment", {})
	model_family = exp.get("model_family")
	ml_configs = project_cfg.get("ml_configs", {})

	if not model_family:
		raise ValueError("experiment.model_family is not specified in the project config")
	model_cfg_path = ml_configs.get(model_family)
	if not model_cfg_path:
		raise ValueError(f"No model config path found for '{model_family}' in ml_configs")

	model_cfg = load_yaml_file(model_cfg_path)

	# Optional: allow project-level overrides (e.g., search_method)
	overrides = {}
	for key in ("search_method",):
		if key in exp:
			overrides[key] = exp[key]

	if overrides:
		model_cfg = deep_update(model_cfg, overrides)

	return model_cfg

