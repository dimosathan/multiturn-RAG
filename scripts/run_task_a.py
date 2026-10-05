#!/usr/bin/env python
"""Task A: rewrite (x5) -> ELSER top-100 -> Cohere rerank (weighted RRF) -> nested RRF -> top-10.

Examples
--------
    # full system on the test set
    python scripts/run_task_a.py --tasks data/test/rag_taskAC.jsonl --out outputs/task_a_test

    # development set (build the task file first with scripts/prepare_dev_tasks.py)
    python scripts/run_task_a.py --tasks data/dev/taskA_dev.jsonl --out outputs/task_a_dev

    # no-rewrite ELSER baseline (Table 4, first row)
    python scripts/run_task_a.py --tasks data/dev/taskA_dev.jsonl --out outputs/elser_baseline \
        --strategies none --no-rerank

    # rewrites only (no Elasticsearch needed)
    python scripts/run_task_a.py --tasks ... --out ... --stop-after rewrite

Every stage is cached in ``--out``; re-running resumes from the last completed
stage.  Outputs: ``fused_top100.jsonl``, ``submission_top10.jsonl`` and
``run_metadata.json`` (config, models, input checksums, code version).
"""

import argparse
import json
from pathlib import Path

from _common import ROOT, cfg, setup

from mtrag.io import load_jsonl
from mtrag.retrieval.pipeline import TaskAPipeline
from mtrag.utils import write_run_metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="task file with conversation `input` (rag_taskAC.jsonl or prepare_dev_tasks.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/task_a.yaml")
    ap.add_argument("--strategies", nargs="+", help="override the strategy list, e.g. `none` or `minimal corpus_specific`")
    ap.add_argument("--no-rerank", action="store_true", help="skip Cohere reranking")
    ap.add_argument("--rerank-method", choices=["rrf", "minmax", "reranker_only"], help="override rerank.method")
    ap.add_argument("--stop-after", choices=["rewrite", "retrieve", "rerank"])
    ap.add_argument("--limit", type=int, help="process only the first N tasks (smoke test)")
    ap.add_argument("--force", action="store_true", help="reuse --out even if it was created with a different config")
    a = ap.parse_args()

    c = cfg(a.config)
    if a.strategies:
        c["strategies"] = a.strategies
    if a.no_rerank:
        c.setdefault("rerank", {})["enabled"] = False
    if a.rerank_method:
        c.setdefault("rerank", {})["method"] = a.rerank_method

    needs_llm = a.strategies != ["none"]
    registry = setup()
    rows = load_jsonl(a.tasks)[: a.limit] if a.limit else load_jsonl(a.tasks)

    retriever = reranker = None
    if a.stop_after != "rewrite":
        from mtrag.retrieval.elser import ElserRetriever
        r = c["retrieval"]
        retriever = ElserRetriever(r["index_map"], r.get("field", "text_embedding"), r.get("model_id", ".elser_model_1"), r.get("doc_id_field", "doc_id"))
        if c.get("rerank", {}).get("enabled", True) and a.stop_after != "retrieve":
            from mtrag.retrieval.rerank import CohereReranker
            reranker = CohereReranker(model=c["rerank"].get("model", "Cohere-rerank-v4.0-pro"))

    meta_path = Path(a.out) / "run_metadata.json"
    if meta_path.exists() and not a.force:
        prev = json.loads(meta_path.read_text(encoding="utf-8"))
        if prev.get("config") != c or prev.get("args", {}).get("limit") != a.limit:
            raise SystemExit(f"{a.out} holds cached stages from a different configuration; "
                             "use a new --out directory (or --force to reuse the cache).")
    write_run_metadata(meta_path, config=c, inputs={"tasks": a.tasks},
                       models={"rewriter": registry.describe().get("rewriter")} if needs_llm else {},
                       args=vars(a), repo_dir=ROOT)
    llm = registry["rewriter"] if needs_llm else None
    out = TaskAPipeline(c, a.out, llm=llm, retriever=retriever, reranker=reranker).run(rows, a.stop_after)
    print(f"Done -> {out}")
    print(registry.tracker.report())


if __name__ == "__main__":
    main()
