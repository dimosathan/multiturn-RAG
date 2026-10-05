"""Minimal chat-completion client for Azure OpenAI / Azure AI Foundry endpoints.

All experiments were run against Azure-hosted models through the REST
``chat/completions`` API with an ``api-key`` header.  This module reproduces
that behaviour (including exponential back-off on HTTP 429) while removing the
hard-coded credentials of the original notebooks: endpoints and keys are read
from environment variables (see ``.env.example``).

Model roles
-----------
Each pipeline stage refers to a *role* (e.g. ``"rewriter"``, ``"generator"``)
that is mapped to a :class:`ModelSpec` in the YAML configuration.  This keeps
model routing (paper, App. C.2) explicit and easy to ablate.
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol

import requests


class ChatModel(Protocol):
    def __call__(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 400) -> str: ...


@dataclass
class ModelSpec:
    """Connection settings for one model deployment.

    ``endpoint_env`` / ``key_env`` name the environment variables that hold the
    full chat-completions URL and the API key.  ``model`` is sent in the payload
    when set (required by Azure AI Foundry serverless models such as
    DeepSeek-V3.2; ignored by Azure OpenAI deployments).
    """

    name: str
    endpoint_env: str
    key_env: str
    model: Optional[str] = None
    price_in: float = 0.0  # USD per 1M input tokens (for cost reporting only)
    price_out: float = 0.0  # USD per 1M output tokens
    max_retries: int = 5
    backoff_base: float = 3.0
    timeout: float = 60.0

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "ModelSpec":
        return cls(name=name, **d)


@dataclass
class UsageTracker:
    """Thread-safe token / cost accounting per model."""

    tokens: Dict[str, list] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, spec: ModelSpec, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            t = self.tokens.setdefault(spec.name, [0, 0, spec.price_in, spec.price_out])
            t[0] += prompt_tokens
            t[1] += completion_tokens

    def cost(self) -> float:
        return sum(i * pi / 1e6 + o * po / 1e6 for i, o, pi, po in self.tokens.values())

    def report(self) -> str:
        rows = [f"{n:<18} in={i:>10,} out={o:>9,}  ${i * pi / 1e6 + o * po / 1e6:.4f}" for n, (i, o, pi, po) in self.tokens.items()]
        rows.append(f"{'TOTAL':<18} ${self.cost():.4f}")
        return "\n".join(rows)


class AzureChatClient:
    """Callable ``(system, user, temperature, max_tokens) -> str`` for one deployment.

    Returns an empty string after ``max_retries`` failures, exactly like the
    original notebooks; downstream stages treat ``""`` as a failed call and
    fall back to their defaults.
    """

    def __init__(self, spec: ModelSpec, tracker: Optional[UsageTracker] = None, session: Optional[requests.Session] = None):
        self.spec = spec
        self.tracker = tracker
        self.session = session or requests.Session()
        self.endpoint = os.environ.get(spec.endpoint_env, "")
        self.key = os.environ.get(spec.key_env, "")
        if not self.endpoint or not self.key:
            raise EnvironmentError(
                f"Model '{spec.name}' needs environment variables {spec.endpoint_env} and {spec.key_env} (see .env.example)."
            )

    def __call__(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 400) -> str:
        payload = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.spec.model:
            payload["model"] = self.spec.model
        headers = {"Content-Type": "application/json", "api-key": self.key}
        for attempt in range(self.spec.max_retries):
            try:
                r = self.session.post(self.endpoint, headers=headers, json=payload, timeout=self.spec.timeout)
                if r.status_code == 200:
                    data = r.json()
                    usage = data.get("usage") or {}
                    if self.tracker:
                        self.tracker.add(self.spec, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                    return (data["choices"][0]["message"]["content"] or "").strip()
                if r.status_code == 429:
                    time.sleep(self.spec.backoff_base * (2**attempt) + random.random())
                    continue
                time.sleep(2)
            except (requests.RequestException, KeyError, ValueError):
                time.sleep(2)
        return ""


class ModelRegistry:
    """Lazily builds one :class:`AzureChatClient` per role from the config."""

    def __init__(self, models: Dict[str, dict], roles: Dict[str, str], tracker: Optional[UsageTracker] = None):
        self.specs = {name: ModelSpec.from_dict(name, d) for name, d in models.items()}
        self.roles = roles
        self.tracker = tracker or UsageTracker()
        self._clients: Dict[str, AzureChatClient] = {}

    def describe(self) -> Dict[str, dict]:
        """Role -> deployment description (no secrets) for run metadata."""
        out = {}
        for role, name in self.roles.items():
            s = self.specs.get(name)
            out[role] = {"name": name, "model": s.model if s else None, "endpoint_env": s.endpoint_env if s else None}
        return out

    def __getitem__(self, role: str) -> ChatModel:
        name = self.roles.get(role, role)
        if name not in self._clients:
            self._clients[name] = AzureChatClient(self.specs[name], self.tracker)
        return self._clients[name]


class FakeChatModel:
    """Deterministic stand-in for tests: returns ``responder(system, user)`` and records calls."""

    def __init__(self, responder: Callable[[str, str], str]):
        self.responder = responder
        self.calls: list = []
        self._lock = threading.Lock()

    def __call__(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 400) -> str:
        with self._lock:
            self.calls.append({"system": system, "user": user, "temperature": temperature, "max_tokens": max_tokens})
        return self.responder(system, user)
