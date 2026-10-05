"""Prompt templates for the Task B / C generation pipeline (paper App. C.1).

Each function reproduces, character for character, the f-string used in the
notebook cell that the paper's Appendix C describes ("FINAL SAFE v2 + GPT-4o",
``TEST_SET_FINAL.ipynb`` cell 33).  Golden-prompt tests guard against drift.
"""

from __future__ import annotations

from typing import Dict, List

from .text import QTYPE_CONFIG, bullet_facts

# (system message, user message) pairs ------------------------------------------------


def conversational(question: str, history: List[str], max_words: int = 50) -> tuple[str, str]:
    hist_block = "\n".join(history[-4:]) if history else ""
    user = f"""Brief friendly response to conversational message.

{f"History: {hist_block}" if hist_block else ""}
User: {question}

Response (under {max_words} words):"""
    return "Friendly assistant.", user


def no_context_refusal(question: str, max_words: int = 25) -> tuple[str, str]:
    user = f"""No information available for this question.

Question: {question}

Short response (under {max_words} words) stating information is not available.
Do NOT say "I don't know" - say "The information is not available" or similar.

Response:"""
    return "Helpful assistant.", user


def span_extraction(question: str, contexts: List[Dict], history: List[str], attempt: int,
                    max_spans: int = 8, n_passages: int = 5) -> tuple[str, str]:
    passages = ""
    for i, ctx in enumerate(contexts[:n_passages], 1):
        passages += f"\n\nPASSAGE {i}:\n{ctx.get('text', '').strip()}"
    hist_block = "\n".join(history[-4:]) if history else ""
    urgency = "⚠️ Answer MUST exist - find it!" if attempt > 1 else ""
    user = f"""Extract sentences answering the question.

{urgency}
{f"History: {hist_block}" if hist_block else ""}
Question: {question}
{passages}

Rules:
1. Copy EXACT sentences
2. Include: names, numbers, dates, key facts
3. Max {max_spans} sentences
4. Prioritize earlier passages

JSON format:
{{"extractedSpans": [{{"passageId": 1, "sentence": "exact text"}}]}}"""
    return "Extract relevant sentences.", user


def generation(spans: List[Dict], question: str, history: List[str], qtype: str = "default",
               base_target: int = 90) -> tuple[str, str]:
    facts = bullet_facts(spans)
    hist_block = "\n".join(history[-4:]) if history else ""
    cfg = QTYPE_CONFIG.get(qtype, QTYPE_CONFIG["default"])
    target_wc = base_target + cfg["length_mod"]
    user = f"""Generate a natural answer using ONLY the facts below.

FACTS:
{facts}

{f"Conversation context: {hist_block}" if hist_block else ""}

Question: {question}
Type: {qtype} - {cfg["hint"]}

CRITICAL RULES:
✅ Use ONLY information from FACTS above
✅ Copy exact phrases for: names, numbers, dates, technical terms
✅ Aim for ~35% verbatim overlap with facts
✅ Make it sound natural and complete
❌ NO outside knowledge
❌ NO hedging (seems, possibly, maybe)
❌ NO meta-phrases (based on, according to)

Length: ~{target_wc} words

Answer:"""
    return "Helpful, accurate assistant.", user


def technical_judge(question: str, spans: List[Dict], a: str, b: str,
                    wc_a: int, wc_b: int, ex_a: float, ex_b: float) -> tuple[str, str]:
    facts = bullet_facts(spans, limit=5, max_chars=120)
    user = f"""Compare two answers for quality.

Question: {question}
Facts: {facts}

A: {a}
   ({wc_a}w, {ex_a:.0%} extractiveness)

B: {b}
   ({wc_b}w, {ex_b:.0%} extractiveness)

Evaluate: Faithfulness, Completeness, Naturalness
Ideal extractiveness: 28-45%

JSON: {{"winner": "A|B", "score_A": 0-10, "score_B": 0-10, "reason": "brief"}}"""
    return "Judge assistant.", user


def user_judge(question: str, history: List[str], a: str, b: str) -> tuple[str, str]:
    hist = " | ".join(history[-3:]) if history else ""
    user = f"""As a user, which answer do you prefer?

{f"Context: {hist}" if hist else ""}
Question: {question}

A: {a}

B: {b}

JSON: {{"preferred": "A|B", "confidence": "HIGH|MEDIUM|LOW"}}"""
    return "User perspective.", user


def force_answer(spans: List[Dict], question: str) -> tuple[str, str]:
    facts = bullet_facts(spans, limit=5)
    user = f"""Answer using ONLY these facts. Do NOT say you cannot answer.

Facts:
{facts}

Question: {question}

Direct answer using the facts:"""
    return "Always answers when facts exist.", user


def micro_adjust(answer: str, reason: str, spans: List[Dict]) -> tuple[str, str]:
    facts = bullet_facts(spans, limit=5)
    user = f"""Fix this answer: {reason}

Facts:
{facts}

Current: {answer}

Fix with minimal changes. Use exact phrases from facts.

Fixed:"""
    return "Editor.", user
