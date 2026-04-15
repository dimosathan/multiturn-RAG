"""
utils/history.py
Conversation history parsing and formatting utilities.
"""

import re
from typing import List, Dict, Tuple


def parse_hybrid_history(text: str) -> Tuple[str, List[Dict]]:
    """
    Parse a raw conversation string into (last_query, history_list).
    Handles both plain-text and structured (User:/Assistant:) formats.

    Returns:
        (query_str, history)  where history is a list of
        {"role": "user"|"assistant", "text": str} dicts,
        or (text, []) if no structure is detected.
    """
    user_re      = re.compile(r"^user[:\.]?\s*(.+)", re.IGNORECASE)
    assistant_re = re.compile(r"^assistant[:\.]?\s*(.+)", re.IGNORECASE)

    lines   = [ln.strip() for ln in text.splitlines() if ln.strip()]
    history = []

    for ln in lines:
        um = user_re.match(ln)
        am = assistant_re.match(ln)
        if um:
            history.append({"role": "user",      "text": um.group(1).strip()})
        elif am:
            history.append({"role": "assistant", "text": am.group(1).strip()})

    if not history:
        return text.strip(), []

    return history[-1]["text"], history[:-1]


def history_to_text(
    history: List[Dict],
    user_max: int = 6,
    assistant_max: int = 3,
) -> str:
    """
    Format a history list into an XML-structured string for the rewriter prompt.
    Includes the last `user_max` user turns and `assistant_max` assistant turns,
    in chronological order.

    Args:
        history:       list of {"role": ..., "text": ...} dicts
        user_max:      max number of user turns to include
        assistant_max: max number of assistant turns to include

    Returns:
        XML-wrapped conversation string.
    """
    output  = ["<conversationhistory>"]
    u_count = 0
    a_count = 0

    for h in reversed(history):
        if h["role"] == "user" and u_count < user_max:
            output.append(f'<turn role="user">{h["text"]}</turn>')
            u_count += 1
        elif h["role"] == "assistant" and a_count < assistant_max:
            output.append(f'<turn role="assistant">{h["text"]}</turn>')
            a_count += 1

    output.append("</conversationhistory>")
    return "\n".join(reversed(output))
