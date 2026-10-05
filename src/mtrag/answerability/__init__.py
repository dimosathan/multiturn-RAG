"""Answerability gates for Task C.

* :class:`MultiJudgeGate` — document / span / answer judges + arbiter (default).
* :class:`SingleClassifierGate` — single-judge baseline.
"""

from .multi_judge import MultiJudgeGate, supreme
from .single import SingleClassifierGate

__all__ = ["MultiJudgeGate", "SingleClassifierGate", "supreme"]
