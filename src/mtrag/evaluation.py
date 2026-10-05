"""Retrieval metrics and significance testing.

The definitions follow the BEIR / ``pytrec_eval`` conventions used by the
official MTRAG evaluation script (``run_retrieval_eval.py``):

* ``Recall@k = |Rel ∩ Ret@k| / |Rel|`` (macro-averaged over queries);
* ``nDCG@k`` with graded gains ``rel`` and ``log2(rank + 1)`` discount, where the
  ideal DCG is computed from *all* judged-relevant documents of the query.

Only queries present in the qrels are scored; queries with no prediction
count as 0.  For leaderboard numbers always use the official script — this
module is meant for fast iteration, ablations and bootstrap tests.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


def recall_at_k(ranking: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    relevant = {d for d, r in rel.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def ndcg_at_k(ranking: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted((r for r in rel.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def per_query(run: Run, qrels: Qrels, metric: str, k: int) -> Dict[str, float]:
    fn = {"recall": recall_at_k, "ndcg": ndcg_at_k}[metric]
    return {q: fn(list(run.get(q, [])), rel, k) for q, rel in qrels.items() if any(r > 0 for r in rel.values())}


def evaluate(run: Run, qrels: Qrels, ks: Iterable[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ks:
        for m in ("recall", "ndcg"):
            vals = per_query(run, qrels, m, k)
            out[f"{m}@{k}"] = sum(vals.values()) / len(vals) if vals else 0.0
    return out


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float], n: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """Two-sided paired bootstrap test on per-query scores (paper App. B, 10,000 resamples).

    Returns the mean difference ``b - a`` and the p-value for H0: no difference
    (fraction of resamples whose mean difference has the opposite sign, doubled).
    """
    qs = sorted(set(a) & set(b))
    if not qs:
        raise ValueError("No common queries.")
    diffs: List[float] = [b[q] - a[q] for q in qs]
    obs = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    m = len(diffs)
    opposite = 0
    for _ in range(n):
        s = sum(diffs[rng.randrange(m)] for _ in range(m)) / m
        if (obs >= 0 and s <= 0) or (obs < 0 and s >= 0):
            opposite += 1
    return {"mean_diff": obs, "p_value": min(1.0, 2 * opposite / n), "n_queries": m}
