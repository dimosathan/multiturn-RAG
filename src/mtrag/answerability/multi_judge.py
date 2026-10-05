"""Multi-judge answerability gate (paper §3 Task C, §5 Table 6; thesis §5.4).

Ported from the "Supreme Court" experiments in ``4.ipynb`` (cells 72-78,
versions v2.0-v2.6); this module implements v2.6.

Pipeline (v2.6)
---------------
0a. *Question expander* — short (<= 6 words) follow-ups are rewritten into a
    full question using the last 5 history lines.
0b. *History sufficiency* — if the previous assistant turns already answer the
    question (confidence >= 0.80) -> ANSWERABLE (early exit).
0c. *New-entity guard* — if the original question introduces proper nouns that
    appear in neither history nor passages and no spans were extracted,
    judges 1 and 2 are pre-set to UNANSWERABLE (documented in the v2.6 header;
    the saved notebook cell is truncated, so the wiring of this step is
    reconstructed from that description).
1.  Span extraction, 2. candidate answer (docs + history).
3.  Judge 1 (documents), Judge 2 (spans), Judge 3 (candidate answer).
4.  Arbiter ("Supreme"): J3 override (J1=J2=UNANS, J3=ANS with conf >= 0.92)
    -> ANSWERABLE; otherwise confidence-boosted majority vote (J2 counts 1.5
    when ANSWERABLE with conf >= 0.90); ties broken by J2.

All calls use GPT-4o at tau = 0 in the experiments.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..generation.text import clean_model_output, word_count
from ..llm import ChatModel
from ..utils import parse_json_object

IDK_INDICATORS = [
    "i don't know", "i do not know",
    "do not have enough information", "don't have enough information",
    "do not have specific information", "don't have specific information",
    "insufficient information", "i cannot answer", "unable to answer",
    "cannot find", "no information available",
    "the information is not available", "not enough information",
]

# Verbatim from 4.ipynb (v2.6).
FEW_SHOT_CALIBRATION = '\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nCALIBRATION EXAMPLES (v2.6):\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nExample 1: ANSWERABLE — Direct from docs\nQuestion: "How many teams are in the NFL?"\nDocuments: ["There are 32 teams in the NFL..."]\nVerdict: ANSWERABLE ✅\n\nExample 2: ANSWERABLE — History contains the answer\nQuestion: "What is it?"\nHistory: [Assistant: "Groundwater contamination refers to harmful substances..."]\nDocuments: [Unrelated docs]\nVerdict: ANSWERABLE ✅ — History already answered it.\n\nExample 3: ANSWERABLE — Feature docs imply worth\nQuestion: "Is it worth having a web chat widget?"\nDocuments: [How to configure/integrate web chat — features described]\nVerdict: ANSWERABLE ✅\n\nExample 4: ANSWERABLE ⚠️ DOMAIN ADJACENCY\nQuestion: "What enforcement options for spousal support violations?"\nHistory: [Child support enforcement: DCSS, police, FPLS, liens...]\nDocuments: [Family law docs]\nVerdict: ANSWERABLE ✅\nReasoning: Child and spousal support enforcement use the same legal\nmechanisms. Domain-adjacent facts from the same legal domain = valid.\nDo NOT say "this is about child support, not spousal support."\n\nExample 5: ANSWERABLE ⚠️ PURPOSE MATCH\nQuestion: "Can I visit the law library to prepare for my oral argument?"\nDocuments: [Oral argument preparation, reviewing legal authorities]\nVerdict: ANSWERABLE ✅\nReasoning: Docs address the PURPOSE (legal research for oral prep).\nThe specific method (library visit) is implied. Docs discuss the\ngoal even if not the exact detail mentioned.\n\nExample 6: ANSWERABLE ⚠️ PARTIAL DATA WITH CAVEAT\nQuestion: "What was India\'s population in the 1940s?"\nAnswer: "Not directly in docs, but historically ~390-400 million."\nVerdict: ANSWERABLE ✅ — Real number present = not IDK.\n\nExample 7: ANSWERABLE — Inference from comparison\nQuestion: "Are pedestrian laws the same as bicycle laws?"\nHistory: [Bicycle laws explained in detail]\nDocuments: [Pedestrian safety rules]\nVerdict: ANSWERABLE ✅\nReasoning: History gives bicycle laws, docs give pedestrian laws.\nTogether they allow a direct comparison = ANSWERABLE.\n\nExample 8: UNANSWERABLE — New entity, no grounding\nQuestion: "Isn\'t this where NASA technology comes in?"\nHistory: [EV battery — no NASA mention]\nDocuments: [EV battery — no NASA mention]\nVerdict: UNANSWERABLE ❌ — NASA not in any context, pure speculation.\n\nExample 9: UNANSWERABLE — Time-sensitive + outdated\nQuestion: "Where do the Cardinals play THIS WEEK?"\nDocuments: ["Week 7, October 22, 2017..."]\nVerdict: UNANSWERABLE ❌\n\nExample 10: UNANSWERABLE — Topic mention only\nQuestion: "Why doesn\'t the government prevent living in flood zones?"\nDocuments: [Flood statistics]\nVerdict: UNANSWERABLE ❌\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nKEY PRINCIPLES (v2.6):\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n✅ ANSWERABLE if ANY of:\n   1. Docs directly answer (even partially).\n   2. History contains the answer (even if docs don\'t).\n   3. Docs describe features of X → "Is X worth it?" = ANSWERABLE.\n   4. Domain-adjacent facts (child→spousal, bicycle→pedestrian, etc.)\n   5. Docs address the PURPOSE/GOAL of the question.\n   6. Answer has caveat BUT provides real numbers/facts.\n   7. Logical inference from docs OR history OR combination.\n\n❌ UNANSWERABLE if ALL of:\n   1. Docs AND history completely off-topic.\n   2. No domain adjacency possible.\n   3. OR time-sensitive + only past data available.\n   4. OR question introduces NEW entities not anywhere in context.\n   5. AND answer is purely "I don\'t know" with no real data.\n\n⚠️ NEVER call UNANSWERABLE because:\n   - The question seems "subjective" if docs have factual info.\n   - The answer has a caveat but provides real data.\n   - The domain is "slightly different" (spousal vs child support, etc.)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'


def is_idk_response(text: str) -> bool:
    """Refusal detector; a caveat accompanied by real data (numbers, estimates) is *not* IDK."""
    t = (text or "").strip().lower()
    if not t:
        return True
    if not any(ind in t for ind in IDK_INDICATORS):
        return False
    has_data = re.search(
        r"\b(\d+[\.,]?\d*\s*(million|billion|thousand|percent|%|km|miles|years|people)?"
        r"|approximately|around|estimated|roughly|about \d)",
        t,
    )
    if has_data:
        return False
    return word_count(text) <= 40


def extract_proper_nouns(text: str) -> set:
    nouns = set()
    for i, w in enumerate(text.split()):
        clean = re.sub(r"[^a-zA-Z]", "", w)
        if len(clean) >= 3 and clean[0].isupper() and i > 0 and clean.upper() != clean:
            nouns.add(clean)
    return nouns


def _exp_note(orig_q: str, expanded_q: str) -> str:
    return f'\n⚠️ Original: "{orig_q}" → Resolved: "{expanded_q}"\n' if expanded_q != orig_q else ""


def supreme(j1: Tuple[str, float], j2: Tuple[str, float], j3: Tuple[str, float],
            j3_override: float = 0.92, j2_boost: float = 0.90) -> Tuple[str, float, str]:
    """Arbiter of v2.6 (pure function, unit-tested)."""
    (j1_v, j1_c), (j2_v, j2_c), (j3_v, j3_c) = j1, j2, j3
    A, U = "ANSWERABLE", "UNANSWERABLE"
    if j3_v == A and j3_c >= j3_override and j1_v == U and j2_v == U:
        return A, j3_c * 0.78, f"J3 override (J1=U, J2=U, J3=A@{j3_override})"
    votes_ans = (
        (1.0 if j1_v == A else 0.0)
        + (1.5 if j2_v == A and j2_c >= j2_boost else 1.0 if j2_v == A else 0.0)
        + (1.0 if j3_v == A else 0.0)
    )
    max_votes = 1.0 + (1.5 if j2_c >= j2_boost else 1.0) + 1.0
    votes_unans = max_votes - votes_ans
    avg = (j1_c + j2_c + j3_c) / 3
    if votes_ans > votes_unans:
        if j1_v == j2_v == j3_v == A:
            return A, avg, "Unanimous ANSWERABLE"
        return A, avg * 0.90, "Majority ANSWERABLE"
    if votes_ans < votes_unans:
        if j1_v == j2_v == j3_v == U:
            return U, avg, "Unanimous UNANSWERABLE"
        return U, avg * 0.85, "Majority UNANSWERABLE"
    if j2_v == A:
        return A, j2_c * 0.80, "Tie → J2 ANSWERABLE"
    return U, j2_c * 0.80, "Tie → J2 UNANSWERABLE"


class MultiJudgeGate:
    def __init__(self, llm: ChatModel, n_passages: int = 5, max_spans: int = 8, max_chars_per_doc: int = 2000,
                 history_threshold: float = 0.80, j3_override: float = 0.92, j2_boost: float = 0.90,
                 guard_confidence: float = 0.90):
        self.llm = llm
        self.n_passages, self.max_spans, self.max_chars = n_passages, max_spans, max_chars_per_doc
        self.history_threshold, self.j3_override, self.j2_boost = history_threshold, j3_override, j2_boost
        # ASSUMPTION: the confidence assigned to the pre-set J1/J2 verdicts is not recorded in the
        # truncated notebook cell; 0.90 is a reconstruction default (see module docstring).
        self.guard_confidence = guard_confidence

    # -- helpers -----------------------------------------------------------
    def _json(self, system: str, user: str, max_tokens: int) -> dict | None:
        return parse_json_object(self.llm(system, user, temperature=0.0, max_tokens=max_tokens))

    def expand_question(self, question: str, history: List[str]) -> str:
        if word_count(question) > 6 or not history:
            return question
        hist_block = "\n".join(history[-5:])
        prompt = f"""A user is in a conversation. Rewrite their short/vague message
as a clear, answerable question using the conversation history.

History:
{hist_block}

User's message: "{question}"

Rules:
- ONE sentence, max 20 words.
- Resolve references ("it","this","they") using history.
- Keep GENERAL — do NOT add specifics not in the history.
- Do NOT introduce new entities absent from history.
- If already clear, return unchanged.

Full question:"""
        expanded = clean_model_output(self.llm("Question expander.", prompt, temperature=0.0, max_tokens=80))
        return expanded if (expanded and len(expanded) >= 5) else question

    def history_sufficiency(self, expanded_q: str, history: List[str]) -> Tuple[bool, str, float]:
        asst = [h for h in history if h.startswith("Assistant:")]
        if not asst:
            return False, "", 0.0
        hist_text = "\n".join(asst[-4:])
        prompt = f"""Does the conversation history already contain sufficient
information to answer this follow-up question?

Previous Assistant responses:
{hist_text}

Follow-up question: "{expanded_q}"

JSON:
{{
  "history_contains_answer": true/false,
  "reason": "brief",
  "confidence": 0.0-1.0
}}"""
        d = self._json("History checker.", prompt, 150)
        if d:
            try:
                conf = float(d.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            if d.get("history_contains_answer", False) and conf >= self.history_threshold:
                return True, d.get("reason", ""), conf
        return False, "", 0.0

    def new_entity_guard(self, question: str, expanded_q: str, history: List[str], contexts: List[Dict], spans: List[Dict]) -> Tuple[bool, str]:
        if question == expanded_q:
            return False, ""
        nouns = extract_proper_nouns(question)
        if not nouns:
            return False, ""
        ctx = (" ".join(history) + " " + " ".join(c.get("text", "") for c in contexts)).lower()
        missing = [n for n in nouns if n.lower() not in ctx]
        if missing and not spans:
            return True, f"New entities not in docs/history: {missing}"
        return False, ""

    def extract_spans(self, expanded_q: str, contexts: List[Dict], history: List[str]) -> List[Dict]:
        passages = "".join(f"\n\n━━ PASSAGE {i} ━━\n{c.get('text', '').strip()[:3000]}" for i, c in enumerate(contexts[: self.n_passages], 1))
        hist_block = "\n".join(history[-3:]) if history else "None"
        prompt = f"""Extract sentences that answer the question.

History: {hist_block}
Question: {expanded_q}

Passages:{passages}

JSON (max {self.max_spans}):
{{"extractedSpans": [{{"passage":1,"text":"sentence"}}]}}"""
        d = self._json("Extractor.", prompt, 800)
        if not d:
            return []
        return [{"text": s["text"]} for s in (d.get("extractedSpans") or [])[: self.max_spans] if isinstance(s, dict) and "text" in s]

    def candidate_answer(self, expanded_q: str, history: List[str], contexts: List[Dict], spans: List[Dict]) -> str:
        facts = ""
        if spans:
            facts += "KEY FACTS:\n" + "".join(f"• {s['text']}\n" for s in spans) + "\n"
        facts += "FULL DOCUMENTS:\n"
        for i, c in enumerate(contexts[: self.n_passages], 1):
            facts += f"\nDoc {i}: {c.get('text', '').strip()[:1500]}\n"
        hist_block = "\n".join(history[-4:]) if history else ""
        hist_section = f"Conversation History (valid source):\n{hist_block}\n" if hist_block else ""
        prompt = f"""Generate a factual answer.

INFORMATION:
{facts}
{hist_section}
Question: {expanded_q}

Instructions:
- Use docs AND history. Domain-adjacent facts are valid.
- Partial info is valid — state caveat but INCLUDE the data.
- For yes/no: clear answer if supported.
- Logical inference from facts is valid.
- Say "not available" ONLY if truly off-topic everywhere.

Answer:"""
        return clean_model_output(self.llm("Answer generator.", prompt, temperature=0.0, max_tokens=500))

    @staticmethod
    def _verdict(d: dict | None, default_v: str) -> Tuple[str, float, str]:
        if not d:
            return default_v, 0.5, "Default"
        try:
            conf = float(d.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return str(d.get("verdict", default_v)).upper(), conf, d.get("analysis", "")

    def judge_documents(self, orig_q, expanded_q, history, contexts):
        docs = "".join(f"\n\n━━ DOC {i} ━━\n{c.get('text', '').strip()[: self.max_chars]}" for i, c in enumerate(contexts[: self.n_passages], 1))
        hist_block = "\n".join(history[-4:]) if history else "None"
        prompt = f"""{FEW_SHOT_CALIBRATION}

EVALUATE:
History: {hist_block}
{_exp_note(orig_q, expanded_q)}
Question: {expanded_q}
Documents:{docs}

STEPS:
1. HISTORY: History already answers? → ANSWERABLE.
2. DOMAIN ADJACENCY: Related domain in history/docs → valid inference.
3. PURPOSE MATCH: Docs address the PURPOSE of the question → ANSWERABLE.
4. TIME SENSITIVITY: "this week/today/now" + past docs → UNANSWERABLE.
5. FEATURE DOCS: "Is X worth it?" + docs describe X's features → ANSWERABLE.
6. DIRECT/PARTIAL: Docs directly answer (even partially)?
7. INFERENCE: Logical inference → ANSWERABLE.

JSON:
{{
  "history_answers": true/false,
  "domain_adjacent": true/false,
  "purpose_matched": true/false,
  "time_sensitive": true/false,
  "directly_or_infers": true/false,
  "analysis": "explanation",
  "verdict": "ANSWERABLE" or "UNANSWERABLE",
  "confidence": 0.0-1.0
}}"""
        return self._verdict(self._json("Judge 1.", prompt, 500), "ANSWERABLE")

    def judge_spans(self, orig_q, expanded_q, history, spans):
        spans_text = "❌ NO SPANS" if not spans else "\n".join(f"• {s.get('text', '')}" for s in spans)
        hist_block = "\n".join(history[-4:]) if history else "None"
        prompt = f"""{FEW_SHOT_CALIBRATION}

EVALUATE:
History: {hist_block}
{_exp_note(orig_q, expanded_q)}
Question: {expanded_q}

Spans:
{spans_text}

STEPS:
1. HISTORY: History answers? → ANSWERABLE even without spans.
2. DOMAIN ADJACENCY: Related domain in history → valid inference.
3. PURPOSE MATCH: Spans address purpose of the question → ANSWERABLE.
4. NO SPANS + no history → UNANSWERABLE.
5. TIME SENSITIVITY: Current question + past spans → UNANSWERABLE.
6. DIRECT: Spans directly answer (even partially)?
7. INFERENCE: Valid logical inference → ANSWERABLE.

JSON:
{{
  "history_answers": true/false,
  "domain_adjacent_or_purpose": true/false,
  "no_spans": true/false,
  "time_sensitive": true/false,
  "spans_answer_or_infer": true/false,
  "analysis": "explanation",
  "verdict": "ANSWERABLE" or "UNANSWERABLE",
  "confidence": 0.0-1.0
}}"""
        return self._verdict(self._json("Judge 2.", prompt, 500), "UNANSWERABLE" if not spans else "ANSWERABLE")

    def judge_answer(self, orig_q, expanded_q, history, answer, j1_analysis=""):
        hist_block = "\n".join(history[-4:]) if history else "None"
        j1_hint = f"\nDoc Judge context: {j1_analysis}\n" if j1_analysis else ""
        prompt = f"""{FEW_SHOT_CALIBRATION}

EVALUATE:
History: {hist_block}
{_exp_note(orig_q, expanded_q)}{j1_hint}
Question: {expanded_q}

Answer:
"{answer}"

STEPS:
1. TIME SENSITIVITY: Current question + outdated answer → UNANSWERABLE.
2. IDK CHECK: Answer PURELY "I don't know" with NO actual data → UNANSWERABLE.
   CRITICAL: Caveat + real data = NOT IDK → ANSWERABLE.
3. DOMAIN ADJACENCY: Answer uses related domain facts → ANSWERABLE.
4. HISTORY GROUNDING: Answer grounded in history → ANSWERABLE.
5. PURPOSE MATCH: Answer addresses the goal/intent → ANSWERABLE.
6. DIRECT: Partial but relevant → ANSWERABLE. Purely off-topic → UNANSWERABLE.

JSON:
{{
  "time_sensitive": true/false,
  "answer_purely_idk": true/false,
  "answer_has_real_data_with_caveat": true/false,
  "domain_adjacent_or_history": true/false,
  "answer_addresses_question": true/false,
  "analysis": "explanation",
  "verdict": "ANSWERABLE" or "UNANSWERABLE",
  "confidence": 0.0-1.0
}}"""
        d = self._json("Judge 3.", prompt, 500)
        if not d:
            return ("UNANSWERABLE" if is_idk_response(answer) else "ANSWERABLE"), 0.5, "Default"
        v, c, a = self._verdict(d, "ANSWERABLE")
        if is_idk_response(answer):
            v, c = "UNANSWERABLE", max(c, 0.85)
        return v, c, a

    # -- driver ------------------------------------------------------------
    def __call__(self, question: str, history: List[str], contexts: List[Dict]) -> Dict:
        contexts = contexts[: self.n_passages]
        expanded = self.expand_question(question, history)
        ok, reason, conf = self.history_sufficiency(expanded, history) if history else (False, "", 0.0)
        if ok:
            answer = self.candidate_answer(expanded, history, contexts, [])
            return {"label": "ANSWERABLE", "confidence": conf, "reason": f"History exit: {reason}",
                    "expanded_question": expanded, "candidate_answer": answer}
        spans = self.extract_spans(expanded, contexts, history)
        answer = self.candidate_answer(expanded, history, contexts, spans)
        guard, guard_reason = self.new_entity_guard(question, expanded, history, contexts, spans)
        if guard:
            j1 = ("UNANSWERABLE", self.guard_confidence, guard_reason)
            j2 = ("UNANSWERABLE", self.guard_confidence, guard_reason)
        else:
            j1 = self.judge_documents(question, expanded, history, contexts)
            j2 = self.judge_spans(question, expanded, history, spans)
        j3 = self.judge_answer(question, expanded, history, answer, j1[2])
        label, conf, reason = supreme(j1[:2], j2[:2], j3[:2], self.j3_override, self.j2_boost)
        return {"label": label, "confidence": conf, "reason": reason, "expanded_question": expanded,
                "candidate_answer": answer, "n_spans": len(spans),
                "judges": {"documents": j1, "spans": j2, "answer": j3}}
