"""Development-set preparation, run metadata and index-building helpers (offline)."""

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

from conftest import ROOT

from mtrag.io import dev_tasks_from_generation, dev_tasks_from_questions, load_jsonl, parse_turn_text, save_jsonl
from mtrag.retrieval.pipeline import TaskAPipeline
from mtrag.retrieval.rewriters import build_rewriter
from mtrag.utils import write_run_metadata

sys.path.insert(0, str(ROOT / "scripts"))


def test_parse_turn_text():
    turns = parse_turn_text("|user|: where do they play\n|assistant|: In Arizona.\ncontinued line\n|user|: since when?")
    assert [t["speaker"] for t in turns] == ["user", "agent", "user"]
    assert turns[1]["text"] == "In Arizona.\ncontinued line"
    assert parse_turn_text("plain query") == [{"speaker": "user", "text": "plain query"}]


def _write_dev_fixture(root: Path):
    for c in ("clapnq", "fiqa", "govt", "cloud"):
        d = root / "retrieval" / c
        d.mkdir(parents=True)
        (d / "dev.tsv").write_text("query-id\tcorpus-id\tscore\n" + (f"conv{c}<::>2\tdoc-{c}\t1\n"), encoding="utf-8")
        save_jsonl([{"_id": f"conv{c}<::>2", "text": "|user|: first q\n|user|: second q"},
                    {"_id": f"conv{c}<::>9", "text": "|user|: not in qrels"}], d / f"{c}_questions.jsonl")
    save_jsonl([{"task_id": "convfiqa<::>2", "Collection": "mt-rag-fiqa-beir-elser-512-100-20240501",
                 "input": [{"speaker": "user", "text": "first q"}, {"speaker": "agent", "text": "ans"}, {"speaker": "user", "text": "second q"}]}],
               root / "generation" / "reference.jsonl")


def test_dev_loaders(tmp_path):
    _write_dev_fixture(tmp_path)
    rows = dev_tasks_from_questions(tmp_path / "retrieval/govt/govt_questions.jsonl", "govt")
    assert rows[0]["Collection"] == "mt-rag-govt-elser-512-100-20240611" and len(rows[0]["input"]) == 2
    gen = dev_tasks_from_generation(tmp_path / "generation/reference.jsonl", keep_ids={"convfiqa<::>2"})
    assert len(gen) == 1 and gen[0]["input"][1]["speaker"] == "agent"


def test_prepare_dev_tasks_script(tmp_path):
    _write_dev_fixture(tmp_path)
    out = tmp_path / "dev"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_dev_tasks.py"), "--data-dir", str(tmp_path), "--out-dir", str(out)],
                       capture_output=True, text=True, check=True)
    rows = load_jsonl(out / "taskA_dev.jsonl")
    assert len(rows) == 4 and "4 kept" in r.stdout          # queries outside the qrels are dropped
    assert (out / "qrels_dev.tsv").read_text().count("\n") == 5  # header + 4 judgments


def test_no_rewrite_baseline_skips_fusion(tmp_path):
    class R:
        def search(self, q, corpus, size=100):
            return [{"document_id": f"{q}-{i}", "score": 10 - i} for i in range(3)]
    rows = [{"task_id": "q1", "Collection": "fiqa", "input": [{"speaker": "user", "text": "a"}, {"speaker": "user", "text": "b"}]}]
    assert build_rewriter("none", None).rewrite("b", [{"speaker": "user", "text": "a"}]) == "b"
    cfg = {"corpora": {}, "strategies": ["none"], "rerank": {"enabled": False}}
    TaskAPipeline(cfg, tmp_path, None, R(), None).run(rows)
    sub = load_jsonl(tmp_path / "submission_top10.jsonl")
    assert [c["document_id"] for c in sub[0]["contexts"]] == ["b-0", "b-1", "b-2"]


def test_run_metadata(tmp_path):
    inp = tmp_path / "tasks.jsonl"
    inp.write_text('{"task_id": 1}\n')
    meta = write_run_metadata(tmp_path / "m.json", config={"a": 1}, inputs={"tasks": inp}, models={"rewriter": {"name": "x"}})
    saved = json.loads((tmp_path / "m.json").read_text())
    assert saved == meta and saved["config"] == {"a": 1} and len(saved["inputs"]["tasks"]["sha256"]) == 64
    assert "mtrag_version" in saved and "timestamp_utc" in saved


def test_build_elser_index_helpers(tmp_path):
    import build_elser_index as B

    es = mock.MagicMock()
    es.indices.exists.return_value = False
    B.ensure_index(es, "fiqa_elser_v1", ".elser_model_1", "text_embedding", "sparse_vector", recreate=False)
    proc = es.ingest.put_pipeline.call_args.kwargs["body"]["processors"][0]["inference"]
    assert proc["model_id"] == ".elser_model_1" and proc["input_output"][0]["output_field"] == "text_embedding"
    mapping = es.indices.create.call_args.kwargs["body"]["mappings"]["properties"]
    assert mapping["text_embedding"]["type"] == "sparse_vector" and mapping["doc_id"]["type"] == "keyword"

    corpus = tmp_path / "c.jsonl"
    save_jsonl([{"_id": "p1", "title": "T", "text": "x"}, {"_id": "p2", "text": "y"}], corpus)
    es.count.return_value = {"count": 0}
    sent = []

    def fake_bulk(client, actions, **kw):
        for a in actions:
            sent.append(a)
            yield True, {}

    with mock.patch("elasticsearch.helpers.streaming_bulk", fake_bulk):
        B.index_corpus(es, "fiqa_elser_v1", str(corpus), with_title=False, chunk=10)
    assert [a["_source"] for a in sent] == [{"doc_id": "p1", "text": "x"}, {"doc_id": "p2", "text": "y"}]
    assert all(a["_op_type"] == "create" for a in sent)
