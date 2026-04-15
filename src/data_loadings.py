import json
from pathlib import Path


def load_jsonl(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_corpus_jsonl(path):
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus[item["doc_id"]] = item
    return corpus


def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, docid, score = line.strip().split()
            qrels.setdefault(qid, {})[docid] = float(score)
    return qrels


def load_queries(path):
    return load_jsonl(path)


def load_predictions(path):
    return load_jsonl(path)
