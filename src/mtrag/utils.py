"""Small shared helpers."""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
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


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def git_commit(repo_dir: str | Path) -> Optional[str]:
    """Current commit hash (with ``-dirty`` suffix if there are local changes), or ``None``."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, timeout=10)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_metadata(path: str | Path, *, config: dict, inputs: dict, models: Optional[dict] = None,
                       args: Optional[dict] = None, repo_dir: str | Path = ".") -> dict:
    """Write ``run_metadata.json``: resolved config, model routing, input checksums,
    code version and environment, so every output can be traced to its settings."""
    from . import __version__

    meta = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "mtrag_version": __version__,
        "git_commit": git_commit(repo_dir),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "args": args or {},
        "inputs": {k: {"path": str(v), "sha256": file_sha256(v)} for k, v in inputs.items()},
        "models": models or {},
        "config": config,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
