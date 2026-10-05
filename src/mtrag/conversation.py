"""Conversation utilities: question/history extraction and history formatting."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Turn = Dict[str, str]


def split_last_turn(input_turns: Sequence[Turn]) -> Tuple[str, List[Turn]]:
    """Return ``(current_query, history)`` where the query is the last turn of ``input``.

    This mirrors the Task A notebooks, which treat ``input[-1]`` as the query
    and ``input[:-1]`` as the history.
    """
    if not input_turns:
        return "", []
    return (input_turns[-1].get("text") or "").strip(), list(input_turns[:-1])


def current_question(input_turns: Sequence[Turn]) -> str:
    """Text of the most recent *user* turn (used by the generation pipelines)."""
    for turn in reversed(input_turns or []):
        if (turn.get("speaker") or "").lower() == "user":
            return (turn.get("text") or "").strip()
    return ""


def history_lines(input_turns: Sequence[Turn], question: str) -> List[str]:
    """``["User: ...", "Assistant: ...", ...]`` up to (excluding) the current question.

    Faithful to the generation notebooks: iteration stops at the first user
    turn whose text equals the current question.
    """
    lines: List[str] = []
    for turn in input_turns or []:
        speaker = (turn.get("speaker") or "").lower()
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        if speaker == "user" and text == question:
            break
        lines.append(("User: " if speaker == "user" else "Assistant: ") + text)
    return lines


def format_history_xml(
    history: Sequence[Turn],
    user_max: int = 6,
    assistant_max: int = 3,
    *,
    eos_marker: bool = True,
    legacy_tag_order: bool = True,
) -> str:
    """XML-structured history used by the query rewriters.

    Turns are selected from the most recent backwards, keeping at most
    ``user_max`` user turns and ``assistant_max`` assistant turns (the two caps
    are independent, e.g. "6U3A").

    ``legacy_tag_order`` reproduces a quirk of the reference notebooks: the
    original implementation reversed the *whole* output list, so the closing
    tag ``</conversation_history>`` appears first and the opening tag last.
    It is kept as the default so that rewrites are reproducible; set it to
    ``False`` for well-formed XML.  ``eos_marker`` appends `` </s>`` to every
    turn (used by all strategies except Anchor-Keyword).
    """
    picked: List[str] = []
    n_user = n_asst = 0
    suffix = " </s>" if eos_marker else ""
    for turn in reversed(history):
        speaker = turn.get("speaker")
        if speaker == "user" and n_user < user_max:
            picked.append(f"  <turn role='user'>{turn['text']}</turn>{suffix}")
            n_user += 1
        elif speaker == "agent" and n_asst < assistant_max:
            picked.append(f"  <turn role='assistant'>{turn['text']}</turn>{suffix}")
            n_asst += 1
    picked.reverse()
    if legacy_tag_order:
        return "\n".join(["</conversation_history>", *picked, "<conversation_history>"])
    return "\n".join(["<conversation_history>", *picked, "</conversation_history>"])
