"""Shared CLI helpers (config loading, model registry)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtrag.llm import ModelRegistry  # noqa: E402
from mtrag.utils import load_env, load_yaml  # noqa: E402


def setup(models_cfg: str = "configs/models.yaml") -> ModelRegistry:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env(ROOT / ".env")
    m = load_yaml(ROOT / models_cfg if not Path(models_cfg).is_absolute() else models_cfg)
    return ModelRegistry(m["models"], m["roles"])


def cfg(path: str) -> dict:
    p = Path(path)
    return load_yaml(p if p.is_absolute() or p.exists() else ROOT / p)
