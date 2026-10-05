#!/usr/bin/env python
"""Build the development-set Task A input (777 answerable/partial queries) and a merged qrels file.

    python scripts/prepare_dev_tasks.py                       # from data/retrieval/*/<corpus>_questions.jsonl
    python scripts/prepare_dev_tasks.py --source generation   # from data/generation/reference.jsonl

``--source questions`` (default) uses the official retrieval query files, the
input of the development experiments; in the MTRAG release they contain the
user turns of each conversation.  ``--source generation`` takes the
conversations from a generation file, which also contains the agent turns.
Only queries that appear in the qrels are kept, so both sources give the same
query set.

Outputs (default ``data/dev/``): ``taskA_dev.jsonl`` (input for
``run_task_a.py``) and ``qrels_dev.tsv`` (input for ``evaluate_retrieval.py``).
"""

import argparse
from collections import Counter
from pathlib import Path

from _common import ROOT  # noqa: F401  (sets sys.path)

from mtrag.io import COLLECTION_IDS, canonical_corpus, dev_tasks_from_generation, dev_tasks_from_questions, load_qrels, save_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--source", choices=["questions", "generation"], default="questions")
    ap.add_argument("--generation-file", default=None, help="default: <data-dir>/generation/reference.jsonl")
    ap.add_argument("--out-dir", default="data/dev")
    a = ap.parse_args()

    data, out = Path(a.data_dir), Path(a.out_dir)
    qrels = {}
    for corpus in COLLECTION_IDS:
        p = data / "retrieval" / corpus / "dev.tsv"
        if not p.exists():
            raise SystemExit(f"missing {p} (see data/README.md)")
        qrels.update(load_qrels(p))

    if a.source == "questions":
        rows = []
        for corpus in COLLECTION_IDS:
            rows += dev_tasks_from_questions(data / "retrieval" / corpus / f"{corpus}_questions.jsonl", corpus)
    else:
        rows = dev_tasks_from_generation(a.generation_file or data / "generation" / "reference.jsonl")

    kept = [r for r in rows if r["task_id"] in qrels]
    missing = sorted(set(qrels) - {r["task_id"] for r in kept})
    save_jsonl(kept, out / "taskA_dev.jsonl")
    with open(out / "qrels_dev.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q, rel in qrels.items():
            for d, s in rel.items():
                f.write(f"{q}\t{d}\t{s}\n")

    per_corpus = Counter(canonical_corpus(r["Collection"]) for r in kept)
    n_agent = sum(any(t.get("speaker") == "agent" for t in r["input"]) for r in kept)
    print(f"source={a.source}: {len(rows)} rows read, {len(kept)} kept (in qrels), {len(qrels)} qrels queries")
    print("per corpus:", dict(per_corpus))
    print(f"tasks whose history contains agent turns: {n_agent}/{len(kept)}")
    if missing:
        print(f"WARNING: {len(missing)} qrels queries have no task row, e.g. {missing[:3]}")
    print(f"wrote {out / 'taskA_dev.jsonl'} and {out / 'qrels_dev.tsv'}")


if __name__ == "__main__":
    main()
