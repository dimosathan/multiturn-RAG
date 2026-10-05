"""Agentic evidence-grounded generation pipeline (paper §3, Task B; App. C).

Stages
------
0. Triage — no passages: conversational reply (regex) or short refusal (<=25 words).
1. Evidence span extraction — up to 8 verbatim sentences from the top-5
   passages; one retry with an urgency prefix; fallback to the top-3 passages.
2. Dual candidate generation — greedy (tau=0.0) and stochastic (tau=0.1),
   question-type-specific length target.
   If both candidates are refusals although evidence exists, a force-answer
   prompt is issued.
3. Judges — technical judge (always) and user-satisfaction judge (sampled,
   60 % of turns).
4. Selection — composite score of :mod:`mtrag.generation.selection`.
5. Micro-adjustment — only if the answer is <50 or >150 words or r4 < 0.28;
   the edit is kept only if it is not a refusal and has >=30 words.
6. Forbidden-phrase removal on the final string.

Model routing (default config): span extraction and technical judge ->
DeepSeek-V3.2; generation -> GPT-4o; user judge, refusals, force-answer and
micro-adjustments -> GPT-4o-mini.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..conversation import current_question, history_lines
from ..llm import ChatModel
from ..utils import parse_json_object
from . import prompts as P
from .selection import ExtractivenessBand, SelectionWeights, selection_scores
from .text import (clean_model_output, detect_question_type, extractiveness, is_conversational, is_pure_idk,
                   remove_forbidden, word_count)


@dataclass
class GenerationConfig:
    n_passages: int = 5
    max_spans: int = 8
    min_spans: int = 1
    span_attempts: int = 2
    temp_a: float = 0.0
    temp_b: float = 0.1
    base_target_words: int = 90
    user_judge_rate: float = 0.60
    seed: int = 42
    min_words: int = 50
    max_words: int = 150
    min_words_after_edit: int = 30
    max_words_refusal: int = 25
    max_words_conversational: int = 50
    enable_span_extraction: bool = True
    enable_dual_generation: bool = True
    enable_technical_judge: bool = True
    enable_user_judge: bool = True
    enable_micro_adjustment: bool = True
    enable_force_answer: bool = True
    band: ExtractivenessBand = field(default_factory=ExtractivenessBand)
    weights: SelectionWeights = field(default_factory=SelectionWeights)

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        d = dict(d or {})
        band = ExtractivenessBand(**d.pop("band", {}))
        weights = SelectionWeights(**d.pop("weights", {}))
        return cls(band=band, weights=weights, **d)


@dataclass
class Models:
    """LLM callables per role (see ``configs/task_b.yaml``)."""

    extractor: ChatModel
    generator: ChatModel
    technical_judge: ChatModel
    user_judge: ChatModel
    light: ChatModel  # refusals, conversational replies, force-answer, micro-adjustments


class GenerationPipeline:
    def __init__(self, models: Models, cfg: GenerationConfig = GenerationConfig()):
        self.m, self.cfg = models, cfg

    # ---------------------------------------------------------------- stage 0
    def conversational_reply(self, question: str, history: List[str]) -> str:
        s, u = P.conversational(question, history, self.cfg.max_words_conversational)
        return clean_model_output(self.m.light(s, u, temperature=0.3, max_tokens=100))

    def refusal(self, question: str) -> str:
        s, u = P.no_context_refusal(question, self.cfg.max_words_refusal)
        return clean_model_output(self.m.light(s, u, temperature=0.0, max_tokens=80))

    # ---------------------------------------------------------------- stage 1
    def extract_spans(self, question: str, contexts: List[Dict], history: List[str]) -> List[Dict]:
        for attempt in range(1, self.cfg.span_attempts + 1):
            s, u = P.span_extraction(question, contexts, history, attempt, self.cfg.max_spans, self.cfg.n_passages)
            obj = parse_json_object(self.m.extractor(s, u, temperature=0.0, max_tokens=700))
            spans = []
            if obj:
                for sp in (obj.get("extractedSpans") or [])[: self.cfg.max_spans]:
                    if isinstance(sp, dict) and "sentence" in sp:
                        spans.append({"text": sp["sentence"], "source": f"p{sp.get('passageId')}"})
            if len(spans) >= self.cfg.min_spans:
                return spans
        return contexts[:3]

    # ---------------------------------------------------------------- stage 2
    def generate(self, spans: List[Dict], question: str, history: List[str], temperature: float, qtype: str) -> str:
        s, u = P.generation(spans, question, history, qtype, self.cfg.base_target_words)
        return clean_model_output(self.m.generator(s, u, temperature=temperature, max_tokens=450))

    # ---------------------------------------------------------------- stage 3
    def technical_judge(self, a: str, b: str, question: str, spans: List[Dict]) -> dict:
        s, u = P.technical_judge(question, spans, a, b, word_count(a), word_count(b),
                                 extractiveness(a, spans), extractiveness(b, spans))
        obj = parse_json_object(self.m.technical_judge(s, u, temperature=0.0, max_tokens=150))
        if obj:
            return {"winner": str(obj.get("winner", "B")).upper(), "score_A": obj.get("score_A", 5),
                    "score_B": obj.get("score_B", 5)}
        return {"winner": "B", "score_A": 5, "score_B": 6}  # default of the original code

    def user_judge(self, a: str, b: str, question: str, history: List[str]) -> dict:
        s, u = P.user_judge(question, history, a, b)
        obj = parse_json_object(self.m.user_judge(s, u, temperature=0.0, max_tokens=100))
        if obj:
            return {"preferred": str(obj.get("preferred", "A")).upper(), "confidence": obj.get("confidence", "MEDIUM")}
        return {"preferred": "A", "confidence": "LOW"}

    def sample_user_judge(self, task_id: str) -> bool:
        """Deterministic per-task Bernoulli(rate) draw.

        The original code drew from the global RNG inside a thread pool, so the
        judged subset depended on thread scheduling.  Seeding per task makes the
        subset reproducible while preserving the expected invocation rate.
        """
        return random.Random(f"{self.cfg.seed}:{task_id}").random() < self.cfg.user_judge_rate

    # ---------------------------------------------------------------- stage 5
    def micro_adjust(self, answer: str, spans: List[Dict]) -> tuple[str, Optional[str]]:
        if is_pure_idk(answer):
            return answer, None
        wc, r4 = word_count(answer), extractiveness(answer, spans)
        if wc < self.cfg.min_words:
            reason = f"Too short ({wc}w)"
        elif wc > self.cfg.max_words:
            reason = f"Too long ({wc}w)"
        elif r4 < self.cfg.band.low:
            reason = f"Low extractiveness ({r4:.0%})"
        else:
            return answer, None
        s, u = P.micro_adjust(answer, reason, spans)
        fixed = clean_model_output(self.m.light(s, u, temperature=0.1, max_tokens=350))
        if fixed and not is_pure_idk(fixed) and word_count(fixed) >= self.cfg.min_words_after_edit:
            return fixed, reason
        return answer, reason

    # ----------------------------------------------------------------- driver
    def answer(self, question: str, history: List[str], contexts: List[Dict], task_id: str = "") -> Dict:
        """Run the pipeline for one turn; returns ``{"text": ..., "trace": {...}}``."""
        c = self.cfg
        contexts = contexts[: c.n_passages]
        qtype = detect_question_type(question)
        trace: Dict = {"qtype": qtype}

        if not contexts:
            if is_conversational(question):
                return {"text": self.conversational_reply(question, history), "trace": {**trace, "route": "conversational"}}
            return {"text": self.refusal(question), "trace": {**trace, "route": "no_context_refusal"}}

        spans = self.extract_spans(question, contexts, history) if c.enable_span_extraction else contexts[:3]
        if not spans:
            spans = contexts[:3]
        trace["n_spans"] = len(spans)

        a = self.generate(spans, question, history, c.temp_a, qtype)
        if not c.enable_dual_generation:
            final = a
        else:
            b = self.generate(spans, question, history, c.temp_b, qtype)
            if c.enable_force_answer and is_pure_idk(a) and is_pure_idk(b):
                s, u = P.force_answer(spans, question)
                forced = clean_model_output(self.m.light(s, u, temperature=0.2, max_tokens=400))
                if forced and not is_pure_idk(forced):
                    # NB: the original returns the forced answer directly (no adjustment / filtering).
                    return {"text": forced, "trace": {**trace, "route": "force_answer"}}
            tech = self.technical_judge(a, b, question, spans) if c.enable_technical_judge else {}
            user = self.user_judge(a, b, question, history) if (c.enable_user_judge and self.sample_user_judge(task_id)) else None
            sel = selection_scores(a, b, spans, tech, user, c.band, c.weights)
            final = a if sel["choice"] == "A" else b
            trace.update({"technical": tech, "user": user, "selection": sel})

        if c.enable_micro_adjustment:
            final, reason = self.micro_adjust(final, spans)
            trace["micro_adjustment"] = reason
        final = remove_forbidden(final.strip())
        trace["route"] = "generated"
        return {"text": final, "trace": trace}

    def process_task(self, task: dict) -> dict:
        """MTRAG task row -> submission row (``predictions=[{"text": ...}]``)."""
        inputs = task.get("input", []) or []
        q = current_question(inputs)
        hist = history_lines(inputs, q)
        ctx = (task.get("contexts") or [])[: self.cfg.n_passages]
        out = self.answer(q, hist, ctx, str(task.get("task_id", "")))
        return {
            "conversation_id": task.get("conversation_id", ""),
            "task_id": task.get("task_id", ""),
            "Collection": task.get("Collection", ""),
            "input": inputs,
            "contexts": ctx if out["trace"].get("route") not in ("conversational", "no_context_refusal") else [],
            "predictions": [{"text": out["text"]}],
            "_trace": out["trace"],
        }
