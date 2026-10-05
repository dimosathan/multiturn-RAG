#!/usr/bin/env python
"""Recall@k / nDCG@k of one or more runs, with optional paired-bootstrap comparison.

    python scripts/evaluate_retrieval.py --qrels data/retrieval/clapnq/dev.tsv \
        --run baseline=outputs/elser_no_rewrite.jsonl --run final=outputs/x/fused_top100.jsonl \
        --compare baseline final --metric recall@5

Use the organisers' ``run_retrieval_eval.py`` for official numbers.
"""

import argparse
import json

from _common import cfg  # noqa: F401

from mtrag.evaluation import evaluate, paired_bootstrap, per_query
from mtrag.io import load_qrels, load_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qrels", required=True, nargs="+", help="one or more qrels TSV files (merged)")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--metric", default="recall@5")
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args()

    qrels = {}
    for p in a.qrels:
        qrels.update(load_qrels(p))
    runs = {n: load_run(p) for n, p in (s.split("=", 1) for s in a.run)}
    for name, run in runs.items():
        print(name, json.dumps({k: round(v, 4) for k, v in evaluate(run, qrels, a.k).items()}))
    if a.compare:
        m, k = a.metric.split("@")
        sa, sb = (per_query(runs[x], qrels, m, int(k)) for x in a.compare)
        print(f"{a.compare[1]} - {a.compare[0]} on {a.metric}:", paired_bootstrap(sa, sb, a.n_boot))


if __name__ == "__main__":
    main()
