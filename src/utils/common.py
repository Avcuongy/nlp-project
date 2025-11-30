from __future__ import annotations

import json
import logging
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
	"""
	Set random seeds for reproducibility across Python, NumPy, and (optionally) PyTorch.

	Args:
		seed: Seed value to set.
		deterministic: If True and PyTorch is available, set deterministic flags.
	"""
	random.seed(seed)
	np.random.seed(seed)

	try:
		import torch  # type: ignore

		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		if deterministic:
			torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
			torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
	except Exception:
		# PyTorch not installed or CUDA not available; silently continue
		pass


def get_device(use_gpu: bool = True) -> str:
	"""
	Return computing device string: "cuda" if available and requested, otherwise "cpu".
	"""
	if not use_gpu:
		return "cpu"
	try:
		import torch  # type: ignore

		return "cuda" if torch.cuda.is_available() else "cpu"
	except Exception:
		return "cpu"


def ensure_dir(path: os.PathLike | str) -> Path:
	"""
	Create directory if it does not exist and return the Path object.
	"""
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
	"""Return a formatted timestamp string."""
	return datetime.now().strftime(fmt)


def save_json(path: os.PathLike | str, obj: Any, *, ensure_ascii: bool = False, indent: int = 2) -> None:
	"""Save a Python object as JSON to the given path."""
	p = Path(path)
	ensure_dir(p.parent)
	with p.open("w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)


def load_json(path: os.PathLike | str) -> Any:
	"""Load JSON content from a file and return the corresponding Python object."""
	with Path(path).open("r", encoding="utf-8") as f:
		return json.load(f)


def save_pickle(path: os.PathLike | str, obj: Any) -> None:
	"""Serialize a Python object to a pickle file."""
	p = Path(path)
	ensure_dir(p.parent)
	with p.open("wb") as f:
		pickle.dump(obj, f)


def load_pickle(path: os.PathLike | str) -> Any:
	"""Load a Python object from a pickle file."""
	with Path(path).open("rb") as f:
		return pickle.load(f)


def read_text(path: os.PathLike | str, encoding: str = "utf-8") -> str:
	"""Read entire text file content."""
	with Path(path).open("r", encoding=encoding) as f:
		return f.read()


def write_text(path: os.PathLike | str, text: str, encoding: str = "utf-8") -> None:
	"""Write text content to a file, creating parent directories if needed."""
	p = Path(path)
	ensure_dir(p.parent)
	with p.open("w", encoding=encoding) as f:
		f.write(text)


def resolve_path(*parts: str | os.PathLike) -> Path:
	"""Join and resolve a filesystem path from multiple parts."""
	return Path(*parts).expanduser().resolve()


_LOGGER_INITIALIZED = False


def get_logger(name: str = "nlp_project", level: int = logging.INFO) -> logging.Logger:
	"""
	Get a configured logger that logs to stdout with a concise formatter.

	The configuration is applied once per process.
	"""
	global _LOGGER_INITIALIZED
	logger = logging.getLogger(name)
	logger.setLevel(level)
	if not _LOGGER_INITIALIZED:
		handler = logging.StreamHandler()
		handler.setLevel(level)
		formatter = logging.Formatter(
			fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S",
		)
		handler.setFormatter(formatter)
		root = logging.getLogger()
		root.handlers.clear()
		root.addHandler(handler)
		root.setLevel(level)
		_LOGGER_INITIALIZED = True
	return logger


def chunked(iterable: Iterable[Any], size: int) -> Iterable[list[Any]]:
	"""Yield successive chunks from an iterable."""
	chunk: list[Any] = []
	for item in iterable:
		chunk.append(item)
		if len(chunk) >= size:
			yield chunk
			chunk = []
	if chunk:
		yield chunk

