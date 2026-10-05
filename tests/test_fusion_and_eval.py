import math

import pytest

from mtrag.conversation import format_history_xml
from mtrag.evaluation import evaluate, ndcg_at_k, paired_bootstrap, recall_at_k
from mtrag.io import canonical_corpus
from mtrag.retrieval.fusion import NESTED_RRF_PARAMS, nested_rrf, weighted_rrf
from mtrag.retrieval.rerank import fuse_minmax, fuse_rrf, minmax


def test_weighted_rrf_formula():
    fused = weighted_rrf({"a": ["x", "y"], "b": ["y", "z"]}, {"a": 1.0, "b": 2.0}, k=10)
    # y: 1/12 + 2/11 ; x: 1/11 ; z: 2/12
    assert fused == ["y", "z", "x"]


def test_nested_rrf_matches_manual_two_level_computation():
    r = {"minimal": ["d1", "d2", "d3"], "corpus_specific": ["d2", "d1", "d4"],
         "cot": ["d4", "d5"], "hyde": ["d5", "d4"], "anchor_keyword": ["d4", "d1"]}
    p = NESTED_RRF_PARAMS["fiqa"]
    weak = weighted_rrf({k: r[k] for k in ("anchor_keyword", "cot", "hyde")}, {"anchor_keyword": .33, "cot": .33, "hyde": .34}, 40)
    expected = weighted_rrf({"minimal": r["minimal"], "corpus_specific": r["corpus_specific"], "weak_consensus": weak}, p["weights"], p["k_final"])
    assert nested_rrf(r, p["k_final"], p["weights"]) == expected
    assert weak[0] == "d4"


def test_table2_parameters_frozen():
    assert NESTED_RRF_PARAMS["clapnq"] == {"k_final": 20, "weights": {"minimal": 0.55, "corpus_specific": 0.40, "weak_consensus": 0.05}}
    assert NESTED_RRF_PARAMS["govt"]["k_final"] == 40 and NESTED_RRF_PARAMS["cloud"]["weights"]["minimal"] == 0.65
    for p in NESTED_RRF_PARAMS.values():
        assert math.isclose(sum(p["weights"].values()), 1.0)


def test_score_fusion():
    assert list(minmax([3, 3])) == [0, 0]
    f = fuse_minmax([10, 5, 0], [0, 1, 0.5])
    assert f.argmax() == 1  # 0.5*0.5 + 0.5*1
    r = fuse_rrf([3, 2, 1], [1, 2, 3], alpha=0.5, k=60)
    assert r[0] == pytest.approx(1 / 61 + 0.5 / 63)


def test_metrics():
    rel = {"a": 1, "b": 1, "c": 2}
    assert recall_at_k(["a", "x", "c"], rel, 2) == pytest.approx(1 / 3)
    assert ndcg_at_k(["c", "a", "b"], rel, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], rel, 2) == 0.0
    res = evaluate({"q": ["a"]}, {"q": rel, "empty": {"z": 0}}, ks=[1])
    assert res["recall@1"] == pytest.approx(1 / 3)  # query without positives is ignored


def test_bootstrap_detects_consistent_gain():
    a = {str(i): 0.0 for i in range(50)}
    b = {str(i): 1.0 if i % 2 else 0.5 for i in range(50)}
    out = paired_bootstrap(a, b, n=500)
    assert out["mean_diff"] > 0 and out["p_value"] < 0.01


def test_canonical_corpus():
    assert canonical_corpus("ibmcloud") == "cloud"
    assert canonical_corpus("mt-rag-fiqa-beir-elser-512-100-20240501") == "fiqa"
    with pytest.raises(KeyError):
        canonical_corpus("wiki")


def test_history_xml_legacy_quirk_and_caps(history_14):
    x = format_history_xml(history_14, 6, 3)
    lines = x.splitlines()
    assert lines[0] == "</conversation_history>" and lines[-1] == "<conversation_history>"
    assert sum("role='user'" in l for l in lines) == 6 and sum("role='assistant'" in l for l in lines) == 3
    good = format_history_xml(history_14, 6, 1, eos_marker=False, legacy_tag_order=False)
    assert good.startswith("<conversation_history>") and "</s>" not in good
