"""The rewriters must send byte-identical prompts to those of the reference notebooks.

Golden fixtures were captured by executing the original notebook classes with
a fake LLM on a synthetic 14-turn history (``tests/fixtures/golden_rewrites.json``).
"""

import pytest
from conftest import load_fixture

from mtrag.llm import FakeChatModel
from mtrag.retrieval.rewriters import build_rewriter

GOLD = load_fixture("golden_rewrites.json")
RESPONSE = '{"class":"non-standalone","rewritten version":"RW","standalone_query":"SQ","hypothetical_passage":"HP","anchors":["A1","a1","A2"],"keywords":["K1"],"reasoning":"R"}'

CASES = [
    ("minimal", "minimal", "clapnq"),
    ("corpus_clapnq", "corpus_specific", "clapnq"),
    ("corpus_fiqa", "corpus_specific", "fiqa"),
    ("corpus_govt", "corpus_specific", "govt"),
    ("corpus_cloud", "corpus_specific", "cloud"),
    ("cot", "cot", "clapnq"),
    ("hyde", "hyde", "clapnq"),
    ("anchor", "anchor_keyword", "clapnq"),
]


@pytest.mark.parametrize("key,strategy,corpus", CASES)
def test_prompts_identical(key, strategy, corpus, history_14):
    llm = FakeChatModel(lambda s, u: RESPONSE)
    out = build_rewriter(strategy, llm).rewrite("final question?", history_14, corpus)
    call = llm.calls[0]
    assert call["system"] == GOLD[key]["system"]
    assert call["user"] == GOLD[key]["user"]
    assert call["max_tokens"] == GOLD[key]["max_tokens"]
    assert call["temperature"] == 0.0
    assert out == GOLD[key]["result"]


@pytest.mark.parametrize("strategy", ["minimal", "corpus_specific", "cot", "hyde", "anchor_keyword"])
def test_first_turn_and_failures_fall_back(strategy, history_14):
    llm = FakeChatModel(lambda s, u: "not json")
    rw = build_rewriter(strategy, llm)
    assert rw.rewrite("q", [], "clapnq") == "q"  # first turn: no LLM call
    assert llm.calls == []
    assert rw.rewrite("q", history_14, "clapnq") == "q"  # unparsable output


def test_anchor_keyword_dedup_and_cap(history_14):
    many = ",".join(f'"kw{i}"' for i in range(20))
    llm = FakeChatModel(lambda s, u: '{"rewritten version":"one two three","anchors":["IBM","ibm","IBM Cloud"],"keywords":[' + many + "]}")
    out = build_rewriter("anchor_keyword", llm).rewrite("q", history_14, "cloud")
    assert out.split()[:5] == ["one", "two", "three", "IBM", "IBM"]  # "ibm" deduplicated
    assert len(out.split()) <= 28
