#!/usr/bin/env python
"""Task C: end-to-end RAG.

Step 1 (retrieval) is Task A; pass its fused run with ``--run``:

    python scripts/run_task_a.py --tasks data/test/rag_taskAC.jsonl --out outputs/task_a_test
    python scripts/run_task_c.py --tasks data/test/rag_taskAC.jsonl \
        --run outputs/task_a_test/submission_top10.jsonl --out outputs/task_c.jsonl
"""

import argparse

from _common import cfg, setup
from run_task_b import build_pipeline, write_outputs

from mtrag.answerability import MultiJudgeGate, SingleClassifierGate
from mtrag.io import attach_passage_text, load_corpus, load_jsonl
from mtrag.rag import RAGPipeline
from mtrag.utils import thread_map


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="task file with conversations (rag_taskAC.jsonl)")
    ap.add_argument("--run", required=True, help="Task A run (task_id, Collection, contexts[document_id, score])")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/task_c.yaml")
    ap.add_argument("--gate", choices=["single", "multi_judge"], help="override answerability.gate")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    registry = setup()
    c = cfg(a.config)
    ca = cfg(c["retrieval_config"])
    corpora = {name: load_corpus(p, title_sep="\n") for name, p in ca["corpora"].items()}
    rows = attach_passage_text(load_jsonl(a.run), load_jsonl(a.tasks), corpora, top_k=10)
    if a.limit:
        rows = rows[: a.limit]

    ans = c.get("answerability", {})
    gate_name = a.gate or ans.get("gate", "multi_judge")
    if gate_name == "single":
        gate = SingleClassifierGate(registry["answerability_single"], ans.get("overlap_threshold", 0.40))
    else:
        gate = MultiJudgeGate(registry["answerability_multi"], n_passages=c.get("n_passages", 3), **ans.get("multi_judge", {}))

    gen = build_pipeline(registry, cfg(c["generation_config"]).get("generation", {}))
    pipe = RAGPipeline(gen, gate, n_passages=c.get("n_passages", 3), refusal_threshold=ans.get("refusal_threshold", 0.70))
    out = thread_map(pipe.process_task, rows, c.get("workers", 5), f"task C [{gate_name}]")
    write_outputs(out, a.out)
    print(registry.tracker.report())


if __name__ == "__main__":
    main()
