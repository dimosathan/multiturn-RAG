"""Cross-encoder reranking and retriever/reranker score fusion (paper §3, Task A, stage 3).

Two fusion rules are implemented:

``rrf`` (default; paper Eq. 1, thesis Eq. 5.2.1)
    ``s(d) = 1 / (k + r_E(d)) + alpha / (k + r_R(d))`` with ``k = 60``, ``alpha = 0.5``.

``minmax`` (score interpolation, used in the development notebooks)
    ``s(d) = (1 - alpha) * minmax(s_ELSER(d)) + alpha * minmax(s_Cohere(d))``.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence

import numpy as np
import requests


class CohereReranker:
    """Cohere Rerank v4 through the Azure AI Foundry REST endpoint.

    Environment: ``AZURE_COHERE_ENDPOINT`` (``.../providers/cohere/v2/rerank``)
    and ``AZURE_COHERE_KEY``.  On failure the reranker returns zeros for all
    documents, which (with min-max fusion) leaves the retriever order intact.
    """

    def __init__(self, model: str = "Cohere-rerank-v4.0-pro", endpoint_env: str = "AZURE_COHERE_ENDPOINT",
                 key_env: str = "AZURE_COHERE_KEY", max_retries: int = 5, timeout: float = 60.0):
        self.model = model
        self.endpoint = os.environ.get(endpoint_env, "")
        self.key = os.environ.get(key_env, "")
        if not self.endpoint or not self.key:
            raise EnvironmentError(f"Set {endpoint_env} and {key_env} to use Cohere reranking.")
        self.max_retries, self.timeout = max_retries, timeout

    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        if not query or not docs:
            return [0.0] * len(docs)
        payload = {"model": self.model, "query": query, "documents": list(docs), "top_n": len(docs)}
        headers = {"Content-Type": "application/json", "api-key": self.key}
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    scores = [0.0] * len(docs)
                    for item in r.json()["results"]:
                        scores[item["index"]] = float(item["relevance_score"])
                    return scores
                if r.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            except (requests.RequestException, KeyError, ValueError):
                time.sleep(2)
        return [0.0] * len(docs)


def minmax(x: Sequence[float]) -> np.ndarray:
    """Min-max normalisation; a constant vector maps to all zeros (as in the original)."""
    a = np.asarray(x, dtype=np.float32)
    if a.size == 0 or a.max() == a.min():
        return np.zeros_like(a)
    return (a - a.min()) / (a.max() - a.min())


def fuse_minmax(retriever_scores: Sequence[float], reranker_scores: Sequence[float], alpha: float = 0.5) -> np.ndarray:
    return (1.0 - alpha) * minmax(retriever_scores) + alpha * minmax(reranker_scores)


def fuse_rrf(retriever_scores: Sequence[float], reranker_scores: Sequence[float], alpha: float = 0.5, k: float = 60.0) -> np.ndarray:
    """Eq. 1 of the paper.  Ranks are derived from the scores (1 = best)."""
    def ranks(s):
        order = np.argsort(-np.asarray(s, dtype=np.float64), kind="stable")
        r = np.empty(len(s), dtype=np.float64)
        r[order] = np.arange(1, len(s) + 1)
        return r
    return 1.0 / (k + ranks(retriever_scores)) + alpha / (k + ranks(reranker_scores))


def rerank_contexts(
    contexts: List[dict],
    query: str,
    doc_text: dict,
    reranker: Optional[CohereReranker],
    method: str = "rrf",
    alpha: float = 0.5,
    k: float = 60.0,
    top_k: int = 60,
) -> List[dict]:
    """Rerank the first ``top_k`` contexts of one query; contexts beyond ``top_k`` keep their order.

    ``contexts`` items are ``{"document_id", "score"}``; ``doc_text`` maps ids to
    passage text (``"N/A"`` is sent for missing ids, as in the original).
    """
    head, tail = contexts[:top_k], contexts[top_k:]
    if not head:
        return contexts
    texts = [doc_text.get(str(c["document_id"]), "N/A") for c in head]
    rer = reranker.score(query, texts) if reranker else [0.0] * len(head)
    ret = [float(c["score"]) for c in head]
    if method == "minmax":
        fused = fuse_minmax(ret, rer, alpha)
    elif method == "rrf":
        fused = fuse_rrf(ret, rer, alpha, k)
    elif method == "reranker_only":
        fused = np.asarray(rer)
    else:
        raise ValueError(f"Unknown fusion method {method!r}")
    new = [{**c, "score": float(s)} for c, s in zip(head, fused)]
    new.sort(key=lambda c: c["score"], reverse=True)
    return new + tail
