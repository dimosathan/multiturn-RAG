"""Multi-strategy query rewriting (paper §3, Task A, stage 1; App. B.13).

Five complementary reformulations are produced for every non-first turn:

==================  ===========================  ===============  ==========
Strategy            Behaviour                    History window   max_tokens
==================  ===========================  ===============  ==========
``minimal``         coreference resolution only  6 user / 3 asst  400
``corpus_specific`` domain-aware (per corpus)    6/3 (FiQA: 6/0)  400
``cot``             reasoning trace + rewrite    6 user / 3 asst  600
``hyde``            standalone + hypothetical    6 user / 3 asst  800
                    2-4 sentence passage
``anchor_keyword``  rewrite + anchors + keywords 6 user / 1 asst  450
==================  ===========================  ===============  ==========

All strategies use DeepSeek-V3.2 at temperature 0.0 (paper §4).
First turns (empty history) are returned unchanged by every strategy, and any
parsing failure falls back to the original query; both behaviours are
inherited from the original implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from ..conversation import Turn, format_history_xml
from ..llm import ChatModel
from ..utils import parse_json_object
from . import prompts as P

STRATEGIES = ("minimal", "corpus_specific", "cot", "hyde", "anchor_keyword")


@dataclass
class Rewriter:
    """Base class: build prompt -> call LLM -> parse -> fallback to the query."""

    llm: ChatModel
    name: str = "base"
    max_tokens: int = 400
    temperature: float = 0.0

    def build(self, query: str, history: Sequence[Turn], corpus: str) -> tuple[str, str]:  # pragma: no cover
        raise NotImplementedError

    def parse(self, raw: str, query: str, corpus: str = "") -> str:
        obj = parse_json_object(raw)
        if not obj:
            return query
        out = obj.get("rewritten version", query)
        return out if isinstance(out, str) and out.strip() else query

    def rewrite(self, query: str, history: Sequence[Turn], corpus: str = "") -> str:
        if not history:
            return query
        system, user = self.build(query, history, corpus)
        try:
            raw = self.llm(system, user, temperature=self.temperature, max_tokens=self.max_tokens)
            return self.parse(raw, query, corpus)
        except Exception:
            return query


def _xml_user(history_xml: str, query: str, suffix: str = "") -> str:
    return P.XML_USER_TEMPLATE.format(history=history_xml, query=query, suffix=suffix)


class MinimalRewriter(Rewriter):
    def __init__(self, llm: ChatModel):
        super().__init__(llm, name="minimal", max_tokens=400)

    def build(self, query, history, corpus):
        return P.MINIMAL_SYSTEM, _xml_user(format_history_xml(history, 6, 3), query)

    def parse(self, raw, query, corpus=""):
        # The minimal rewriter did *not* strip / empty-check the output.
        obj = parse_json_object(raw)
        return obj.get("rewritten version", query) if obj else query


class CorpusSpecificRewriter(Rewriter):
    """Per-domain prompts (Table 29).  FiQA uses user turns only (6U0A) and a plain-text frame."""

    def __init__(self, llm: ChatModel):
        super().__init__(llm, name="corpus_specific", max_tokens=400)

    def build(self, query, history, corpus):
        if corpus == "fiqa":
            hist = format_history_xml(history, 6, 0)
            return P.CORPUS_FIQA_SYSTEM, P.FIQA_USER_TEMPLATE.format(history=hist, query=query)
        system = {
            "govt": P.CORPUS_GOVT_SYSTEM,
            "cloud": P.CORPUS_CLOUD_SYSTEM,
        }.get(corpus, P.CORPUS_CLAPNQ_SYSTEM)  # ClapNQ prompt is also the fallback
        return system, _xml_user(format_history_xml(history, 6, 3), query)

    def parse(self, raw, query, corpus=""):
        # Post-processing differed slightly per corpus in the original code:
        # ClapNQ returned the raw value, FiQA stripped it, Govt/Cloud stripped
        # it and fell back to the query when empty.
        obj = parse_json_object(raw)
        if not obj:
            return query
        out = obj.get("rewritten version", query)
        if corpus in ("govt", "cloud"):
            out = str(out).strip()
            return out if out else query
        if corpus == "fiqa":
            return str(out).strip()
        return out


class CoTRewriter(Rewriter):
    """Chain-of-Thought: the ``reasoning`` field is discarded, only the rewrite is used."""

    def __init__(self, llm: ChatModel):
        super().__init__(llm, name="cot", max_tokens=600)

    def build(self, query, history, corpus):
        return P.COT_SYSTEM, _xml_user(format_history_xml(history, 6, 3), query, " with reasoning")

    def parse(self, raw, query, corpus=""):
        obj = parse_json_object(raw)
        return obj.get("rewritten version", query) if obj else query


class HyDERewriter(Rewriter):
    """HyDE: ``standalone_query + " " + hypothetical_passage`` is submitted to the retriever."""

    def __init__(self, llm: ChatModel):
        super().__init__(llm, name="hyde", max_tokens=800)

    def build(self, query, history, corpus):
        return P.HYDE_SYSTEM, _xml_user(format_history_xml(history, 6, 3), query)

    def parse(self, raw, query, corpus=""):
        obj = parse_json_object(raw)
        if not obj:
            return query
        standalone = obj.get("standalone_query", query)
        passage = obj.get("hypothetical_passage", "")
        return f"{standalone} {passage}".strip()


def _norm_term(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"[\"'`]", "", s)


def dedupe_terms(terms: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for t in terms:
        if not t:
            continue
        key = _norm_term(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return out


def cap_words(text: str, max_words: int) -> str:
    w = text.split()
    return text.strip() if len(w) <= max_words else " ".join(w[:max_words]).strip()


class AnchorKeywordRewriter(Rewriter):
    """Rewrite + anchors (<=8) + keywords (<=12), concatenated and capped at 28 words."""

    def __init__(self, llm: ChatModel, max_anchors: int = 8, max_keywords: int = 12, max_words: int = 28):
        super().__init__(llm, name="anchor_keyword", max_tokens=450)
        self.max_anchors, self.max_keywords, self.max_words = max_anchors, max_keywords, max_words

    def build(self, query, history, corpus):
        hist = format_history_xml(history, 6, 1, eos_marker=False, legacy_tag_order=False)
        system = P.anchor_keyword_system(self.max_anchors, self.max_keywords, self.max_words)
        return system, _xml_user(hist, query)

    def parse(self, raw, query, corpus=""):
        obj = parse_json_object(raw)
        if not obj:
            return query
        base = (obj.get("rewritten version") or query).strip()
        anchors = obj.get("anchors") or []
        keywords = obj.get("keywords") or []
        anchors = dedupe_terms([str(x) for x in anchors] if isinstance(anchors, list) else [])[: self.max_anchors]
        keywords = dedupe_terms([str(x) for x in keywords] if isinstance(keywords, list) else [])[: self.max_keywords]
        combined = cap_words(" ".join([base] + anchors + keywords).strip(), self.max_words)
        return combined or query


class NoRewriter(Rewriter):
    """Baseline: the last user turn is sent to the retriever unchanged (paper Table 4, "No rewriting")."""

    def __init__(self, llm: ChatModel = None):
        super().__init__(llm, name="none")

    def rewrite(self, query: str, history: Sequence[Turn], corpus: str = "") -> str:
        return query


REWRITER_CLASSES: Dict[str, Callable[[ChatModel], Rewriter]] = {
    "none": NoRewriter,
    "minimal": MinimalRewriter,
    "corpus_specific": CorpusSpecificRewriter,
    "cot": CoTRewriter,
    "hyde": HyDERewriter,
    "anchor_keyword": AnchorKeywordRewriter,
}


def build_rewriter(strategy: str, llm: ChatModel) -> Rewriter:
    try:
        return REWRITER_CLASSES[strategy](llm)
    except KeyError as e:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {sorted(REWRITER_CLASSES)}") from e
