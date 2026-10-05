"""Candidate selection (paper App. C.3, Eqs. 5-6).

    Score(y) = w_T * T(y) + 5c * U(y) + phi(r4(y, S)) - 2.0 * 1[forbidden(y)]

* ``T(y)`` ∈ [0, 10] — technical-judge score, ``w_T = 0.35``.
* ``U(y)`` — user-satisfaction preference; ``c`` ∈ {1.0, 0.7, 0.4} for
  HIGH / MEDIUM / LOW confidence.  When the user judge is not invoked a
  constant prior of +2.0 is added to candidate A (the greedy one).
* ``phi`` — piecewise extractiveness shaping (Eq. 5).
Ties favour A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .text import contains_forbidden, extractiveness


@dataclass(frozen=True)
class ExtractivenessBand:
    low: float = 0.28    # below -> under-extractive
    ideal: float = 0.38  # [low, ideal] -> ideal
    high: float = 0.50   # (ideal, high] -> acceptable; above -> over-extractive
    r_under: float = -1.5
    r_ideal: float = 2.5
    r_accept: float = 1.5
    r_over: float = 0.5

    def phi(self, r: float) -> float:
        if self.low <= r <= self.ideal:
            return self.r_ideal
        if self.ideal < r <= self.high:
            return self.r_accept
        if r < self.low:
            return self.r_under
        return self.r_over


CONFIDENCE_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}


@dataclass
class SelectionWeights:
    technical: float = 0.35
    user: float = 5.0
    prior_a: float = 2.0
    forbidden_penalty: float = 2.0


def selection_scores(a: str, b: str, spans, tech: Optional[dict], user: Optional[dict],
                     band: ExtractivenessBand = ExtractivenessBand(),
                     w: SelectionWeights = SelectionWeights()) -> dict:
    """Return ``{"A": score, "B": score, "r4_A": .., "r4_B": .., "choice": "A"|"B"}``."""
    tech = tech or {}
    s = {"A": 0.0, "B": 0.0}
    s["A"] += _num(tech.get("score_A", 5)) * w.technical
    s["B"] += _num(tech.get("score_B", 5)) * w.technical
    if user:
        conf = CONFIDENCE_WEIGHT.get(str(user.get("confidence", "MEDIUM")).upper(), 0.7)
        s["A" if user.get("preferred", "A") == "A" else "B"] += w.user * conf
    else:
        s["A"] += w.prior_a
    ra, rb = extractiveness(a, spans), extractiveness(b, spans)
    s["A"] += band.phi(ra)
    s["B"] += band.phi(rb)
    if contains_forbidden(a):
        s["A"] -= w.forbidden_penalty
    if contains_forbidden(b):
        s["B"] -= w.forbidden_penalty
    return {**s, "r4_A": ra, "r4_B": rb, "choice": "A" if s["A"] >= s["B"] else "B"}


def _num(x, default: float = 5.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default
