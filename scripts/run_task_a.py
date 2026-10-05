#!/usr/bin/env python
"""Task A: rewrite (x5) -> ELSER top-100 -> Cohere hybrid rerank -> nested RRF -> top-10.

Example
-------
    python scripts/run_task_a.py --tasks data/test/rag_taskAC.jsonl --out outputs/task_a_test
    python scripts/run_task_a.py --tasks ... --out ... --stop-after rewrite   # rewrites only

Intermediate files are cached in ``--out``; re-running resumes from the last
completed stage.  The submission file is ``<out>/submission_top10.jsonl``.
"""

import argparse

from _common import cfg, setup

from mtrag.io import load_jsonl
from mtrag.retrieval.pipeline import TaskAPipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="MTRAG task file with conversation `input` (e.g. rag_taskAC.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/task_a.yaml")
    ap.add_argument("--stop-after", choices=["rewrite", "retrieve", "rerank"])
    ap.add_argument("--limit", type=int, help="process only the first N tasks (smoke test)")
    a = ap.parse_args()

    registry = setup()
    c = cfg(a.config)
    rows = load_jsonl(a.tasks)[: a.limit] if a.limit else load_jsonl(a.tasks)

    retriever = reranker = None
    if a.stop_after != "rewrite":
        from mtrag.retrieval.elser import ElserRetriever
        r = c["retrieval"]
        retriever = ElserRetriever(r["index_map"], r.get("field", "text_embedding"), r.get("model_id", ".elser_model_1"), r.get("doc_id_field", "doc_id"))
        if c.get("rerank", {}).get("enabled", True) and a.stop_after != "retrieve":
            from mtrag.retrieval.rerank import CohereReranker
            reranker = CohereReranker(model=c["rerank"].get("model", "Cohere-rerank-v4.0-pro"))

    out = TaskAPipeline(c, a.out, llm=registry["rewriter"], retriever=retriever, reranker=reranker).run(rows, a.stop_after)
    print(f"Done -> {out}")
    print(registry.tracker.report())


if __name__ == "__main__":
    main()
