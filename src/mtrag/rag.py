"""Task C: end-to-end RAG = Task A retrieval -> answerability gate -> Task B generation.

Turns judged UNANSWERABLE with confidence >= 0.70 receive a short refusal
(<= 25 words, paper App. D.1); all other turns are routed through
:class:`mtrag.generation.GenerationPipeline` using the top-3 retrieved
passages (paper App. D.3).
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .conversation import current_question, history_lines
from .generation.pipeline import GenerationPipeline
from .generation.text import is_conversational

Gate = Callable[[str, List[str], List[Dict]], Dict]


class RAGPipeline:
    def __init__(self, generator: GenerationPipeline, gate: Gate, n_passages: int = 3, refusal_threshold: float = 0.70):
        self.gen, self.gate, self.n_passages = generator, gate, n_passages
        self.refusal_threshold = refusal_threshold

    def process_task(self, task: dict) -> dict:
        inputs = task.get("input", []) or []
        q = current_question(inputs)
        hist = history_lines(inputs, q)
        ctx = (task.get("contexts") or [])[: self.n_passages]
        trace: Dict = {}
        if is_conversational(q):
            text = self.gen.conversational_reply(q, hist)
            trace["route"] = "conversational"
        else:
            verdict = self.gate(q, hist, ctx)
            trace["answerability"] = {k: v for k, v in verdict.items() if k != "candidate_answer"}
            conf = verdict.get("confidence")
            conf = conf if isinstance(conf, (int, float)) else 1.0  # categorical gates carry no score
            if verdict["label"] == "UNANSWERABLE" and conf >= self.refusal_threshold:
                text = self.gen.refusal(q)
                trace["route"] = "refusal"
            else:
                out = self.gen.answer(q, hist, ctx, str(task.get("task_id", "")))
                text = out["text"]
                trace.update(out["trace"])
        return {
            "conversation_id": task.get("conversation_id", ""),
            "task_id": task.get("task_id", ""),
            "Collection": task.get("Collection", ""),
            "input": inputs,
            "contexts": ctx,
            "predictions": [{"text": text}],
            "_trace": trace,
        }
