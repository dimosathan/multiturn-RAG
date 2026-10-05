"""Data loading / saving helpers for the MTRAG benchmark files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Union

PathLike = Union[str, Path]

# Short corpus name -> official collection identifier used by the evaluation scripts.
COLLECTION_IDS: Dict[str, str] = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "govt": "mt-rag-govt-elser-512-100-20240611",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
}

# Every spelling of a collection that appears in the dev / test files.
_ALIASES: Dict[str, str] = {
    "clapnq": "clapnq",
    "fiqa": "fiqa",
    "govt": "govt",
    "cloud": "cloud",
    "ibmcloud": "cloud",
    **{v: k for k, v in COLLECTION_IDS.items()},
}


def canonical_corpus(collection: str) -> str:
    """Map any collection spelling (``ibmcloud``, full MTRAG id, ...) to a short corpus name.

    >>> canonical_corpus("mt-rag-ibmcloud-elser-512-100-20240502")
    'cloud'
    """
    key = (collection or "").strip()
    if key in _ALIASES:
        return _ALIASES[key]
    low = key.lower()
    for short in ("clapnq", "fiqa", "govt"):
        if short in low:
            return short
    if "cloud" in low:
        return "cloud"
    raise KeyError(f"Unknown collection: {collection!r}")


def iter_jsonl(path: PathLike) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: PathLike) -> List[dict]:
    return list(iter_jsonl(path))


def save_jsonl(rows: Iterable[dict], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_corpus(path: PathLike, max_chars: int | None = None, title_sep: str = " ") -> Dict[str, str]:
    """Load a passage-level corpus as ``{passage_id: "title<sep>text"}``.

    The passage id is read from ``_id`` / ``id`` / ``document_id`` (the MTRAG
    corpora use ``_id``).  ``max_chars`` truncates each passage; the reranking
    stage uses ``max_chars=2000`` and ``title_sep=" "``,
    while the Task C input preparation used ``title_sep="\\n"`` and no truncation.
    """
    corpus: Dict[str, str] = {}
    for row in iter_jsonl(path):
        pid = row.get("_id") or row.get("id") or row.get("document_id")
        if pid is None:
            continue
        title = row.get("title", "") or ""
        text = row.get("text") or row.get("contents") or ""
        full = f"{title}{title_sep}{text}".strip() if title else text.strip()
        corpus[str(pid)] = full[:max_chars] if max_chars else full
    return corpus


def load_qrels(path: PathLike) -> Dict[str, Dict[str, int]]:
    """Load a TREC-style qrels TSV (``query-id  corpus-id  score``, optional header)."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t") if "\t" in line else line.strip().split()
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[-2], parts[-1]
            try:
                rel = int(float(score))
            except ValueError:  # header line
                continue
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def load_run(path: PathLike) -> Dict[str, List[str]]:
    """Load a retrieval prediction file as ``{task_id: [doc_id, ...]}`` (rank order)."""
    run: Dict[str, List[str]] = {}
    for row in iter_jsonl(path):
        tid = str(row.get("task_id") or row.get("taskid"))
        ctxs = row.get("contexts", [])
        run[tid] = [str(c.get("document_id") or c.get("documentid")) for c in ctxs]
    return run


def attach_passage_text(run_rows: List[dict], task_rows: List[dict], corpora: Dict[str, Dict[str, str]], top_k: int = 10) -> List[dict]:
    """Build Task C generation inputs: conversation ``input`` + top-k contexts with passage text.

    ``corpora`` maps short corpus name -> {passage_id: text} (load with
    ``load_corpus(path, title_sep="\\n")`` as in the reference notebook).
    """
    tasks = {str(t["task_id"]): t for t in task_rows}
    out = []
    for r in run_rows:
        tid = str(r["task_id"])
        t = tasks.get(tid)
        if t is None:
            continue
        corpus = corpora.get(canonical_corpus(r["Collection"]), {})
        ctx = [{"document_id": c["document_id"], "text": corpus.get(str(c["document_id"]), ""), "score": c.get("score", 0.0),
                "source": "", "title": ""} for c in r.get("contexts", [])[:top_k]]
        row = {"task_id": tid, "Collection": r["Collection"], "input": t.get("input", []), "contexts": ctx}
        for key in ("conversation_id", "task_type", "turn"):
            if key in t:
                row[key] = t[key]
        out.append(row)
    return out
