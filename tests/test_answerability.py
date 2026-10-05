import pytest
from conftest import load_fixture

from mtrag.answerability import MultiJudgeGate, SingleClassifierGate, supreme
from mtrag.llm import FakeChatModel

GOLD = load_fixture("golden_answerability.json")
HIST = ["User: Tell me about EV batteries", "Assistant: EV batteries last long.", "User: what about cost?", "Assistant: They cost money."]
CTX = [{"text": "Doc one text."}, {"text": "Doc two text."}]
RESP = '{"verdict":"ANSWERABLE","confidence":0.8,"analysis":"ok","history_contains_answer":false,"confidence":0.8,"extractedSpans":[{"passage":1,"text":"S1"}]}'


@pytest.mark.parametrize("row", GOLD["supreme_grid"])
def test_supreme_matches_original(row):
    v1, c1, v2, c2, v3, c3 = row["in"]
    label, conf, reason = supreme((v1, c1), (v2, c2), (v3, c3))
    assert [label, reason] == [row["out"][0], row["out"][2]]
    assert conf == pytest.approx(row["out"][1])


def _gate():
    fake = FakeChatModel(lambda s, u: RESP)
    return MultiJudgeGate(fake), fake


@pytest.mark.parametrize("method,args,key", [
    ("expand_question", ("and NASA?", HIST), "expand"),
    ("extract_spans", ("What about NASA?", CTX, HIST), "spans"),
    ("candidate_answer", ("What about NASA?", HIST, CTX, [{"text": "S1"}]), "answer"),
    ("judge_documents", ("and NASA?", "What about NASA?", HIST, CTX), "j1"),
    ("judge_spans", ("and NASA?", "What about NASA?", HIST, []), "j2"),
    ("judge_answer", ("and NASA?", "What about NASA?", HIST, "Answer text", "j1 analysis"), "j3"),
])
def test_multi_judge_prompts_identical(method, args, key):
    gate, fake = _gate()
    out = getattr(gate, method)(*args)
    gold_calls = GOLD[f"{key}_calls"]
    assert [(c["system"], c["user"], c["max_tokens"]) for c in fake.calls] == [(c["system"], c["user"], c["max_tokens"]) for c in gold_calls]
    gold = GOLD[key]
    assert (list(out) if isinstance(out, tuple) else out) == gold


def test_history_check_prompt_identical():
    gate, fake = _gate()
    assert list(gate.history_sufficiency("What about NASA?", HIST)) == GOLD["hist"]
    assert fake.calls[0]["user"] == GOLD["hist_calls"][0]["user"]


def test_single_gate_matches_original():
    fake = FakeChatModel(lambda s, u: '{"answerability":"UNANSWERABLE","confidence":"HIGH"}')
    gate = SingleClassifierGate(fake)
    low = gate("Where do the Cardinals play this week?", HIST, [{"text": "Unrelated stuff about cooking pasta."}])
    assert low["label"] == GOLD["single_low_overlap"]
    assert (fake.calls[0]["system"], fake.calls[0]["user"]) == (GOLD["single_calls"][0]["system"], GOLD["single_calls"][0]["user"])
    high = gate("Where do the Cardinals play?", HIST, [{"text": "The Cardinals play at Busch Stadium where games happen."}])
    assert high["label"] == GOLD["single_high_overlap"]
    assert len(fake.calls) == 1  # overlap shortcut: no LLM call
    assert gate("q", HIST, [])["label"] == "UNANSWERABLE"
