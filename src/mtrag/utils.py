"""Small shared helpers."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, TypeVar

import yaml

T = TypeVar("T")
R = TypeVar("R")

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> Optional[dict]:
    """Extract the outermost ``{...}`` block from an LLM response and parse it.

    Matches the greedy-regex strategy used throughout the original notebooks.
    Returns ``None`` if no valid JSON object is found.
    """
    m = _JSON_OBJ.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def thread_map(fn: Callable[[T], R], items: Iterable[T], workers: int = 5, desc: str = "") -> List[R]:
    """Order-preserving parallel map with an optional progress bar."""
    items = list(items)
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        it = ex.map(fn, items)
        if tqdm is not None and desc:
            it = tqdm(it, total=len(items), desc=desc)
        return list(it)


def load_env(dotenv_path: str | Path = ".env") -> None:
    """Load ``.env`` if python-dotenv is installed (no-op otherwise)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(dotenv_path)


def deep_get(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
