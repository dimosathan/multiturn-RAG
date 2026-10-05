#!/usr/bin/env python
"""Task B: generation with reference passages.

    python scripts/run_task_b.py --tasks data/test/reference_taskB.jsonl --out outputs/task_b.jsonl

Writes the submission file (official schema) and ``<out>.trace.jsonl`` with
per-turn diagnostics (spans, judge scores, selection, micro-adjustments).
"""

import argparse
from pathlib import Path

from _common import ROOT, cfg, setup

from mtrag.generation import GenerationConfig, GenerationPipeline, Models
from mtrag.io import load_jsonl, save_jsonl
from mtrag.utils import thread_map, write_run_metadata


def build_pipeline(registry, gen_cfg: dict) -> GenerationPipeline:
    models = Models(extractor=registry["extractor"], generator=registry["generator"],
                    technical_judge=registry["technical_judge"], user_judge=registry["user_judge"], light=registry["light"])
    return GenerationPipeline(models, GenerationConfig.from_dict(gen_cfg))


def write_outputs(rows, out: str):
    save_jsonl([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], out)
    save_jsonl([{"task_id": r["task_id"], **r.get("_trace", {})} for r in rows], Path(out).with_suffix(".trace.jsonl"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/task_b.yaml")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    registry = setup()
    c = cfg(a.config)
    pipe = build_pipeline(registry, c.get("generation", {}))
    tasks = load_jsonl(a.tasks)[: a.limit] if a.limit else load_jsonl(a.tasks)
    write_run_metadata(Path(a.out).with_suffix(".meta.json"), config=c, inputs={"tasks": a.tasks},
                       models=registry.describe(), args=vars(a), repo_dir=ROOT)
    rows = thread_map(pipe.process_task, tasks, c.get("workers", 5), "task B")
    write_outputs(rows, a.out)
    print(registry.tracker.report())


if __name__ == "__main__":
    main()
