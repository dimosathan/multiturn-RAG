"""Text utilities for the generation pipeline (extractiveness, IDK detection, question type)."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

# App. C.4 — removed from the final output and penalised (-2.0) during selection.
FORBIDDEN_PHRASES = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i'm uncertain", "i cannot say", "it's unclear", "it is unclear",
    "i cannot answer", "unable to answer", "cannot find information",
]

IDK_INDICATORS = [
    "i don't know", "i do not know", "do not have enough information",
    "don't have enough information", "do not have specific information",
    "don't have specific information", "insufficient information",
    "i cannot answer", "unable to answer", "cannot find",
    "no information available", "information is not available",
]


def word_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])


def clean_model_output(text: str) -> str:
    """Strip wrapping quotes, a leading ``Assistant:``/``Agent:`` tag and repeated whitespace."""
    t = (text or "").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    t = re.sub(r"^(Agent|Assistant):\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def is_pure_idk(text: str, max_words: int = 35) -> bool:
    """A short response (<= ``max_words``) that contains a refusal indicator."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    return any(ind in low for ind in IDK_INDICATORS) and word_count(t) <= max_words


def contains_forbidden(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in FORBIDDEN_PHRASES)


def remove_forbidden(text: str) -> str:
    """Case-insensitive removal of :data:`FORBIDDEN_PHRASES`, then whitespace/punctuation cleanup."""
    out = text
    for phrase in FORBIDDEN_PHRASES:
        out = re.compile(re.escape(phrase), re.IGNORECASE).sub("", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return out.strip()


def ngrams(text: str, n: int) -> set:
    words = (text or "").lower().split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def extractiveness(answer: str, evidence: Iterable[Dict], n: int = 4) -> float:
    """r_n(y, S) = |ngrams(y) ∩ ngrams(concat(S))| / |ngrams(y)|  (paper Eq. 4, n = 4).

    ``evidence`` is a list of ``{"text": ...}`` dicts (extracted spans or passages).
    Tokenisation is lower-cased whitespace splitting, as in the original code.
    """
    a = ngrams(answer, n)
    if not a:
        return 0.0
    e = ngrams(" ".join(s.get("text", "") for s in evidence), n)
    return len(a & e) / len(a)


# App. C.5 / Table 32 — rule-based question type (first match wins).
QTYPE_CONFIG: Dict[str, Dict] = {
    "factoid": {"hint": "Direct answer with specific fact.", "length_mod": -5},
    "explanation": {"hint": "Clear explanation.", "length_mod": +10},
    "how_to": {"hint": "Step-by-step.", "length_mod": +5},
    "summarization": {"hint": "Comprehensive summary.", "length_mod": +15},
    "comparative": {"hint": "Compare systematically.", "length_mod": +5},
    "keyword": {"hint": "Interpret and answer.", "length_mod": 0},
    "default": {"hint": "Complete answer.", "length_mod": 0},
}


def detect_question_type(question: str) -> str:
    q = question.lower().strip()
    if len(q.split()) <= 3 and "?" not in question:
        return "keyword"
    if re.search(r"\bhow (do|can|should|to)\b", q):
        return "how_to"
    if re.search(r"\b(why|explain|reason)\b", q):
        return "explanation"
    if re.search(r"\b(compare|difference|vs|better)\b", q):
        return "comparative"
    if re.search(r"\b(summar|overview)\b", q):
        return "summarization"
    if re.search(r"\b(what|who|when|where|how many|how much)\b", q):
        return "factoid"
    return "default"


_CONVERSATIONAL = [
    r"^\s*thanks?\s*[.!]?\s*$", r"^\s*thank you\s*[.!]?\s*$",
    r"^\s*ok(ay)?\s*[.!]?\s*$", r"^\s*yes\s*[.!?]?\s*$",
    r"^\s*no\s*[.!?]?\s*$", r"^\s*got it\s*[.!]?\s*$",
    r"^\s*(great|cool|nice|alright)\s*[.!]?\s*$",
]


def is_conversational(question: str) -> bool:
    q = (question or "").lower().strip()
    return any(re.search(p, q) for p in _CONVERSATIONAL)


def bullet_facts(spans: List[Dict], limit: int | None = None, max_chars: int | None = None) -> str:
    items = spans if limit is None else spans[:limit]
    return "\n".join(f"• {s.get('text', '')[:max_chars] if max_chars else s.get('text', '')}" for s in items)
