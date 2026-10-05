"""Single-judge answerability gate (baseline of paper Table 39 / thesis Table 6.17).

Decision rule (``TEST_SET_FINAL.ipynb``, cell 25, ``predict_answerability``):

1. No passages -> UNANSWERABLE.
2. Keyword-overlap shortcut: if > 40 % of the question's salient tokens
   (capitalised words, numbers, words of >= 4 characters) occur in the
   passages -> ANSWERABLE without an LLM call.
3. Otherwise an LLM (DeepSeek-V3.2, tau=0) labels the turn
   ANSWERABLE / PARTIAL / UNANSWERABLE with a confidence; an UNANSWERABLE
   label with LOW confidence is flipped to ANSWERABLE (explicit bias towards
   answering).
4. Any failure -> ANSWERABLE.
"""

from __future__ import annotations

import re
from typing import Dict, List

from ..llm import ChatModel
from ..utils import parse_json_object

SYSTEM = "You judge answerability. BIAS towards ANSWERABLE."


def keyword_overlap(question: str, contexts: List[Dict]) -> float | None:
    kws = set(re.findall(r"\b[A-Z][a-z]+|\b\d+|\b\w{4,}\b", question))
    kws = {k.lower() for k in kws if len(k) > 2}
    if not kws:
        return None
    text = " ".join((c.get("text", "") or "").lower() for c in contexts)
    return sum(1 for k in kws if k in text) / len(kws)


def build_prompt(question: str, contexts: List[Dict], history: List[str]) -> str:
    passages = "DOCUMENTS:\n"
    for i, c in enumerate(contexts, 1):
        passages += f"[PASSAGE {i}]\n{(c.get('text', '') or '').strip()}\n\n"
    hist = ("CONVERSATION HISTORY:\n" + "\n".join(history[-2:]).strip() + "\n\n") if history else ""
    return f"""ANSWERABILITY JUDGMENT:

ANSWERABLE: Docs contain relevant facts (even partial)
PARTIAL: Docs contain some info but missing key details
UNANSWERABLE: Temporal mismatch OR completely missing info

Bias: When uncertain → ANSWERABLE

{passages}
{hist}QUESTION: {question}

Return JSON: {{"answerability": "ANSWERABLE"|"PARTIAL"|"UNANSWERABLE", "confidence": "HIGH"|"MEDIUM"|"LOW"}}"""


class SingleClassifierGate:
    def __init__(self, llm: ChatModel, overlap_threshold: float = 0.40):
        self.llm, self.overlap_threshold = llm, overlap_threshold

    def __call__(self, question: str, history: List[str], contexts: List[Dict]) -> Dict:
        if not contexts:
            return {"label": "UNANSWERABLE", "reason": "no_contexts"}
        ov = keyword_overlap(question, contexts)
        if ov is not None and ov > self.overlap_threshold:
            return {"label": "ANSWERABLE", "reason": f"keyword_overlap={ov:.2f}"}
        obj = parse_json_object(self.llm(SYSTEM, build_prompt(question, contexts, history), temperature=0.0, max_tokens=150))
        if obj:
            label = str(obj.get("answerability", "")).strip().upper()
            conf = str(obj.get("confidence", "MEDIUM")).strip().upper()
            if label == "UNANSWERABLE" and conf == "LOW":
                return {"label": "ANSWERABLE", "reason": "low_confidence_flip"}
            if label in {"ANSWERABLE", "PARTIAL", "UNANSWERABLE"}:
                return {"label": label, "confidence": conf, "reason": "llm"}
        return {"label": "ANSWERABLE", "reason": "fallback"}
