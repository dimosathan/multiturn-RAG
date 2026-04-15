"""
utils/io.py
Shared I/O helpers for loading corpora, queries, qrels, predictions, and rewrites.
"""

import os
import csv
import json
from typing import Dict, List, Set


def load_jsonl(path: str) -> List[dict]:
    """Load a JSONL file and return a list of dicts."""
    data = []
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_json(path: str) -> dict | list:
    """Load a JSON file."""
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[dict], path: str) -> None:
    """Save a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_corpus(path: str, max_chars: int = 2000) -> Dict[str, str]:
    """
    Load a passage-level corpus JSONL into a {doc_id: text} dict.
    Truncates text to max_chars for reranking use.
    """
    corpus = {}
    if not os.path.exists(path):
        print(f"[WARNING] Corpus not found: {path}")
        return corpus
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            did = str(j.get("id") or j.get("_id"))
            text = (j.get("title", "") + " " + (j.get("text") or j.get("contents") or "")).strip()
            corpus[did] = text[:max_chars]
    return corpus


def load_queries(path: str) -> Dict[str, dict]:
    """
    Load a queries JSONL file.
    Returns {query_id: {"query": str, "history": list}} dict.
    Handles both single-turn (last turn) and multi-turn (with history) formats.
    """
    from utils.history import parse_hybrid_history

    qmap = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            qid = str(j.get("id") or j.get("_id") or j.get("taskid"))
            query, hist = parse_hybrid_history(j.get("text", ""))
            qmap[qid] = {"query": query, "history": hist}
    return qmap


def load_qrels(path: str) -> Dict[str, Set[str]]:
    """
    Load a qrels TSV file (query_id, doc_id, relevance).
    Returns {query_id: set of relevant doc_ids} for relevance > 0.
    """
    qrels = {}
    if not os.path.exists(path):
        print(f"[WARNING] Qrels not found: {path}")
        return qrels
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            next(reader)  # skip header
        except StopIteration:
            return qrels
        for row in reader:
            if len(row) < 3:
                continue
            qid, doc_id, score = row[0], row[1], int(row[2])
            if score > 0:
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(doc_id)
    return qrels


def load_rewrites(path: str) -> Dict[str, str]:
    """
    Load a rewrites file (JSON list or dict) and return {query_id: rewritten_query}.
    Handles both list format (from rewriting pipeline) and plain dict format.
    """
    if not os.path.exists(path):
        print(f"[WARNING] Rewrites file not found: {path}")
        return {}

    data = load_json(path)
    rewrites_map = {}

    if isinstance(data, list):
        for item in data:
            qid = str(item.get("qid") or item.get("taskid"))
            rewrite = item.get("rewritten_query") or item.get("rewrite")
            if qid and rewrite:
                rewrites_map[qid] = rewrite
    elif isinstance(data, dict):
        rewrites_map = {str(k): str(v) for k, v in data.items()}

    print(f"[INFO] Loaded {len(rewrites_map)} rewrites from {os.path.basename(path)}")
    return rewrites_map
