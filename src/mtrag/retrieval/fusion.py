"""Reciprocal Rank Fusion and the two-level *nested* RRF (paper §3, Eq. 2; App. B.9).

Nested RRF
----------
Level 1 pre-aggregates the three high-variance strategies (HyDE, CoT,
Anchor-Keyword) into a single *Weak Consensus* ranking (``k_internal = 40``,
weights 0.33/0.33/0.34).  Level 2 fuses that ranking with the two stable
strategies (Minimal, Corpus-Specific) using corpus-specific weights and
``k_final`` (Table 2 / Table 24).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

Ranking = List[str]  # doc ids, best first


def weighted_rrf(
    rankings: Mapping[str, Ranking],
    weights: Optional[Mapping[str, float]] = None,
    k: float = 60.0,
) -> Ranking:
    """Weighted RRF over named rankings: ``score(d) = sum_s w_s / (k + rank_s(d))``, ranks 1-based.

    Ties are broken by first appearance (Python's sort is stable and dict
    insertion order follows the order in which documents were first seen),
    which matches the original implementation.
    """
    weights = weights or {}
    scores: Dict[str, float] = {}
    for name, ranking in rankings.items():
        w = weights.get(name, 1.0)
        for rank, did in enumerate(ranking, 1):
            scores[did] = scores.get(did, 0.0) + w * (1.0 / (k + rank))
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# Default configuration (dev-tuned, frozen for the test set; paper Table 2).
WEAK_STRATEGIES = ("anchor_keyword", "cot", "hyde")
WEAK_INTERNAL_WEIGHTS = {"anchor_keyword": 0.33, "cot": 0.33, "hyde": 0.34}
K_INTERNAL = 40

NESTED_RRF_PARAMS: Dict[str, dict] = {
    "clapnq": {"k_final": 20, "weights": {"minimal": 0.55, "corpus_specific": 0.40, "weak_consensus": 0.05}},
    "fiqa": {"k_final": 60, "weights": {"minimal": 0.45, "corpus_specific": 0.40, "weak_consensus": 0.15}},
    "govt": {"k_final": 40, "weights": {"minimal": 0.65, "corpus_specific": 0.25, "weak_consensus": 0.10}},
    "cloud": {"k_final": 20, "weights": {"minimal": 0.65, "corpus_specific": 0.30, "weak_consensus": 0.05}},
}


def nested_rrf(
    strategy_rankings: Mapping[str, Ranking],
    k_final: float,
    final_weights: Mapping[str, float],
    weak_strategies: Sequence[str] = WEAK_STRATEGIES,
    weak_weights: Mapping[str, float] = WEAK_INTERNAL_WEIGHTS,
    k_internal: float = K_INTERNAL,
) -> Ranking:
    """Two-level fusion for a single query.

    ``strategy_rankings`` must contain ``minimal`` and ``corpus_specific``;
    weak strategies that are missing are simply skipped.
    """
    weak = {s: strategy_rankings[s] for s in weak_strategies if s in strategy_rankings}
    level2 = {
        "minimal": strategy_rankings.get("minimal", []),
        "corpus_specific": strategy_rankings.get("corpus_specific", []),
    }
    if weak:
        level2["weak_consensus"] = weighted_rrf(weak, weak_weights, k_internal)
    return weighted_rrf(level2, final_weights, k_final)


def nested_rrf_run(
    runs: Mapping[str, Mapping[str, Ranking]],
    corpus_of: Mapping[str, str],
    params: Mapping[str, dict] = NESTED_RRF_PARAMS,
    **kwargs,
) -> Dict[str, Ranking]:
    """Apply :func:`nested_rrf` to every query.

    ``runs`` maps strategy -> {task_id -> ranking}; ``corpus_of`` maps
    task_id -> short corpus name (selects the Table 2 parameters).
    Query ids are taken from the ``minimal`` run, as in the original code.
    """
    out: Dict[str, Ranking] = {}
    for qid in runs["minimal"]:
        p = params[corpus_of[qid]]
        per_q = {s: r.get(qid, []) for s, r in runs.items()}
        out[qid] = nested_rrf(per_q, p["k_final"], p["weights"], **kwargs)
    return out


def flat_rrf_run(runs: Mapping[str, Mapping[str, Ranking]], k: float = 60.0, weights=None) -> Dict[str, Ranking]:
    """Flat (single-level) RRF baseline over all strategies (App. B.9 comparison)."""
    return {qid: weighted_rrf({s: r.get(qid, []) for s, r in runs.items()}, weights, k) for qid in runs["minimal"]}
