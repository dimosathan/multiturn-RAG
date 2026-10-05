#!/usr/bin/env python
"""Offline nested / flat RRF over existing per-strategy run files (no API calls).

    python scripts/fuse_runs.py \
        --run minimal=outputs/x/reranked/minimal.jsonl \
        --run corpus_specific=outputs/x/reranked/corpus_specific.jsonl \
        --run cot=... --run hyde=... --run anchor_keyword=... \
        --out outputs/x/fused_top100.jsonl [--flat] [--uniform]

``--uniform`` uses equal Level-2 weights for every corpus (App. B.9 robustness check).
"""

import argparse

from _common import cfg  # noqa: F401  (sets sys.path)

from mtrag.io import canonical_corpus, iter_jsonl, load_run, save_jsonl
from mtrag.retrieval.fusion import NESTED_RRF_PARAMS, flat_rrf_run, nested_rrf_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="STRATEGY=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flat", action="store_true", help="flat RRF (k=60) instead of nested")
    ap.add_argument("--uniform", action="store_true", help="uniform Level-2 weights")
    ap.add_argument("--keep", type=int, default=100)
    a = ap.parse_args()

    paths = dict(s.split("=", 1) for s in a.run)
    runs = {s: load_run(p) for s, p in paths.items()}
    coll = {str(r["task_id"]): r["Collection"] for r in iter_jsonl(paths["minimal"])}
    if a.flat:
        fused = flat_rrf_run(runs)
    else:
        params = NESTED_RRF_PARAMS
        if a.uniform:
            params = {c: {"k_final": p["k_final"], "weights": {"minimal": 1 / 3, "corpus_specific": 1 / 3, "weak_consensus": 1 / 3}}
                      for c, p in NESTED_RRF_PARAMS.items()}
        fused = nested_rrf_run(runs, {q: canonical_corpus(c) for q, c in coll.items()}, params)
    save_jsonl([{"task_id": q, "Collection": coll[q], "contexts": [{"document_id": d, "score": 1.0 / (i + 1)} for i, d in enumerate(docs[: a.keep])]}
                for q, docs in fused.items()], a.out)
    print(f"wrote {len(fused)} queries -> {a.out}")


if __name__ == "__main__":
    main()
