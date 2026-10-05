"""End-to-end Task A pipeline with on-disk caching of every intermediate stage.

    tasks ──rewrite (x5)──► rewrites/<strategy>.json
          ──ELSER top-100─► runs/<strategy>.jsonl
          ──rerank (RRF)─► reranked/<strategy>.jsonl
          ──nested RRF───► fused_top100.jsonl ──► submission_top10.jsonl

Each stage is skipped when its output already exists, so expensive LLM /
Elasticsearch calls are never repeated and partial runs can be resumed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..conversation import split_last_turn
from ..io import canonical_corpus, load_corpus, load_json, load_jsonl, save_json, save_jsonl
from ..utils import thread_map
from .fusion import NESTED_RRF_PARAMS, nested_rrf_run
from .rewriters import STRATEGIES, build_rewriter

log = logging.getLogger(__name__)


def prepare_tasks(rows: List[dict]) -> List[dict]:
    """Normalise MTRAG task rows to ``{task_id, collection, corpus, query, history}``."""
    tasks = []
    for r in rows:
        query, history = split_last_turn(r.get("input", []))
        collection = r.get("Collection") or r.get("collection")
        tasks.append({
            "task_id": str(r["task_id"]),
            "collection": collection,
            "corpus": canonical_corpus(collection),
            "query": query,
            "history": history,
        })
    return tasks


class TaskAPipeline:
    def __init__(self, cfg: dict, out_dir: str | Path, llm=None, retriever=None, reranker=None):
        self.cfg, self.out = cfg, Path(out_dir)
        self.llm, self.retriever, self.reranker = llm, retriever, reranker
        self.strategies = cfg.get("strategies", list(STRATEGIES))
        self.workers = cfg.get("workers", 5)

    # ------------------------------------------------------------------ stages
    def rewrite(self, tasks: List[dict]) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for strat in self.strategies:
            path = self.out / "rewrites" / f"{strat}.json"
            if path.exists():
                out[strat] = {r["task_id"]: r["rewritten_query"] for r in load_json(path)}
                continue
            rw = build_rewriter(strat, self.llm)
            results = thread_map(lambda t: rw.rewrite(t["query"], t["history"], t["corpus"]), tasks, self.workers, f"rewrite[{strat}]")
            save_json([{"task_id": t["task_id"], "original_query": t["query"], "rewritten_query": q} for t, q in zip(tasks, results)], path)
            out[strat] = {t["task_id"]: q for t, q in zip(tasks, results)}
        return out

    def retrieve(self, tasks: List[dict], rewrites: Dict[str, Dict[str, str]]) -> Dict[str, Path]:
        k = self.cfg.get("retrieval", {}).get("top_k", 100)
        paths = {}
        for strat in self.strategies:
            path = self.out / "runs" / f"{strat}.jsonl"
            paths[strat] = path
            if path.exists():
                continue
            rows = thread_map(
                lambda t: {"task_id": t["task_id"], "Collection": t["collection"],
                           "contexts": self.retriever.search(rewrites[strat][t["task_id"]], t["corpus"], k)},
                tasks, self.workers, f"elser[{strat}]")
            save_jsonl(rows, path)
        return paths

    def rerank(self, tasks: List[dict], rewrites, run_paths: Dict[str, Path]) -> Dict[str, Path]:
        from .rerank import rerank_contexts

        rc = self.cfg.get("rerank", {})
        if not rc.get("enabled", True):
            return run_paths
        corpora: Dict[str, dict] = {}
        for corpus, p in self.cfg["corpora"].items():
            corpora[corpus] = load_corpus(p, max_chars=rc.get("doc_max_chars", 2000), title_sep=" ")
        corpus_of = {t["task_id"]: t["corpus"] for t in tasks}
        paths = {}
        for strat, run_path in run_paths.items():
            path = self.out / "reranked" / f"{strat}.jsonl"
            paths[strat] = path
            if path.exists():
                continue
            rows = load_jsonl(run_path)

            def _one(row):
                tid = str(row["task_id"])
                # The rerank query is the strategy's own rewrite (HyDE: rewrite + passage).
                q = rewrites[strat].get(tid, tid)
                ctx = rerank_contexts(row["contexts"], q, corpora[corpus_of[tid]], self.reranker,
                                      method=rc.get("method", "rrf"), alpha=rc.get("alpha", 0.5),
                                      k=rc.get("k", 60), top_k=rc.get("top_k", 60))
                return {**row, "contexts": ctx}

            save_jsonl(thread_map(_one, rows, self.workers, f"rerank[{strat}]"), path)
        return paths

    def fuse(self, tasks: List[dict], paths: Dict[str, Path]) -> Path:
        from ..io import load_run

        runs = {s: load_run(p) for s, p in paths.items()}
        corpus_of = {t["task_id"]: t["corpus"] for t in tasks}
        coll_of = {t["task_id"]: t["collection"] for t in tasks}
        if len(runs) == 1:  # single strategy (e.g. the no-rewrite baseline): no fusion
            fused = next(iter(runs.values()))
        else:
            params = self.cfg.get("fusion", {}).get("params", NESTED_RRF_PARAMS)
            fused = nested_rrf_run(runs, corpus_of, params)
        keep = self.cfg.get("fusion", {}).get("keep", 100)
        rows = [{"task_id": q, "Collection": coll_of[q],
                 "contexts": [{"document_id": d, "score": 1.0 / (i + 1)} for i, d in enumerate(docs[:keep])]}
                for q, docs in fused.items()]
        p100 = self.out / "fused_top100.jsonl"
        save_jsonl(rows, p100)
        top = self.cfg.get("submission_top_k", 10)
        save_jsonl([{**r, "contexts": r["contexts"][:top]} for r in rows], self.out / "submission_top10.jsonl")
        return p100

    # -------------------------------------------------------------------- run
    def run(self, rows: List[dict], stop_after: Optional[str] = None) -> Path:
        tasks = prepare_tasks(rows)
        rewrites = self.rewrite(tasks)
        if stop_after == "rewrite":
            return self.out / "rewrites"
        runs = self.retrieve(tasks, rewrites)
        if stop_after == "retrieve":
            return self.out / "runs"
        reranked = self.rerank(tasks, rewrites, runs)
        if stop_after == "rerank":
            return self.out / "reranked"
        return self.fuse(tasks, reranked)
