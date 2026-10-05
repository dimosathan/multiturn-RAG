"""End-to-end smoke tests of Task A and Task C with fake LLM / retriever / reranker (no network)."""

import json

from mtrag.answerability import SingleClassifierGate
from mtrag.generation import GenerationConfig, GenerationPipeline, Models
from mtrag.io import load_jsonl, save_jsonl
from mtrag.llm import FakeChatModel
from mtrag.rag import RAGPipeline
from mtrag.retrieval.pipeline import TaskAPipeline


class FakeRetriever:
    def search(self, query, corpus, size=100):
        # deterministic, query-dependent ranking
        h = sum(map(ord, query))
        ids = [f"{corpus}-{(h + i) % 7}" for i in range(5)]
        return [{"document_id": d, "score": 10.0 - i} for i, d in enumerate(dict.fromkeys(ids))]


class FakeReranker:
    def score(self, query, docs):
        return [float(len(d)) for d in docs]


def test_task_a_pipeline(tmp_path):
    corpus = tmp_path / "c.jsonl"
    save_jsonl([{"_id": f"clapnq-{i}", "title": "t", "text": "x" * i} for i in range(7)], corpus)
    cfg = {"corpora": {c: str(corpus) for c in ("clapnq", "fiqa", "govt", "cloud")}, "workers": 2}
    rows = [{"task_id": f"q{i}", "Collection": "mt-rag-clapnq-elser-512-100-20240503",
             "input": [{"speaker": "user", "text": "hi"}, {"speaker": "agent", "text": "hello"}, {"speaker": "user", "text": f"question {i}"}]}
            for i in range(3)]
    llm = FakeChatModel(lambda s, u: '{"rewritten version":"rw","standalone_query":"sq","hypothetical_passage":"hp"}')
    TaskAPipeline(cfg, tmp_path / "out", llm, FakeRetriever(), FakeReranker()).run(rows)
    sub = load_jsonl(tmp_path / "out" / "submission_top10.jsonl")
    assert len(sub) == 3 and all(1 <= len(r["contexts"]) <= 10 for r in sub)
    n_calls = len(llm.calls)
    TaskAPipeline(cfg, tmp_path / "out", llm, FakeRetriever(), FakeReranker()).run(rows)  # cached
    assert len(llm.calls) == n_calls


def test_task_c_pipeline_routes_refusals():
    def resp(s, u):
        if s.startswith("You judge answerability"):
            return '{"answerability":"UNANSWERABLE","confidence":"HIGH"}'
        if s.startswith("Extract"):
            return json.dumps({"extractedSpans": [{"passageId": 1, "sentence": "Fact one about pasta."}]})
        return "The information is not available."
    fake = FakeChatModel(resp)
    gen = GenerationPipeline(Models(fake, fake, fake, fake, fake), GenerationConfig())
    rag = RAGPipeline(gen, SingleClassifierGate(fake))
    task = {"task_id": "t", "Collection": "fiqa", "input": [{"speaker": "user", "text": "Where do the Cardinals play this week?"}],
            "contexts": [{"document_id": "d", "text": "Cooking pasta requires boiling water."}]}
    out = rag.process_task(task)
    assert out["_trace"]["route"] == "refusal"
    assert out["predictions"][0]["text"] == "The information is not available."


def test_task_c_refusal_threshold():
    fake = FakeChatModel(lambda s, u: json.dumps({"extractedSpans": [{"passageId": 1, "sentence": "Fact one."}]}) if s.startswith("Extract") else "Some answer text.")
    gen = GenerationPipeline(Models(fake, fake, fake, fake, fake), GenerationConfig(enable_dual_generation=False, enable_micro_adjustment=False))
    task = {"task_id": "t", "Collection": "fiqa", "input": [{"speaker": "user", "text": "Is it worth it?"}],
            "contexts": [{"document_id": "d", "text": "Fact one."}]}
    low = RAGPipeline(gen, lambda q, h, c: {"label": "UNANSWERABLE", "confidence": 0.6}).process_task(task)
    high = RAGPipeline(gen, lambda q, h, c: {"label": "UNANSWERABLE", "confidence": 0.8}).process_task(task)
    assert low["_trace"]["route"] == "generated" and high["_trace"]["route"] == "refusal"
