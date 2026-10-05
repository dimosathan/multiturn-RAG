#!/usr/bin/env python
"""Create the ELSER v1 Elasticsearch indices used by Task A.

    python scripts/build_elser_index.py --deploy-model             # all four corpora
    python scripts/build_elser_index.py --corpora fiqa --recreate  # rebuild one index

For every corpus in ``configs/task_a.yaml`` the script

1. (optionally) downloads and deploys the ELSER model (``--deploy-model``);
2. creates an ingest pipeline whose inference processor writes the ELSER
   expansion of ``text`` into ``text_embedding``;
3. creates the index (``doc_id`` keyword, ``text`` text, ``text_embedding``
   ``sparse_vector``) with that pipeline as default;
4. bulk-indexes the passage-level corpus (``_id`` = ``doc_id`` = passage id,
   ``text`` = passage text, as in the reference notebooks).

Index, field and model names are read from ``configs/task_a.yaml`` so that the
retriever and the index always agree.  Connection settings come from ``.env``
(``ELASTICSEARCH_CLOUD_ID`` + ``ELASTICSEARCH_PASSWORD`` or ``ELASTICSEARCH_URL``).
Elasticsearch >= 8.11 is required for ``sparse_vector``; use
``--field-type rank_features`` on 8.8-8.10.  Ingest runs the model on every
passage, so indexing ClapNQ (~183k passages) takes hours on a small ML node;
the script resumes when re-run.
"""

import argparse
import time

from _common import ROOT, cfg  # noqa: F401  (sets sys.path)

from mtrag.io import iter_jsonl
from mtrag.retrieval.elser import ElserRetriever
from mtrag.utils import load_env


def deploy_model(es, model_id: str, allocations: int, timeout: int = 900):
    try:
        es.ml.get_trained_models(model_id=model_id)
    except Exception:
        es.ml.put_trained_model(model_id=model_id, input={"field_names": ["text_field"]})
    t0 = time.time()
    while time.time() - t0 < timeout:
        info = es.ml.get_trained_models(model_id=model_id, include="definition_status")
        if info["trained_model_configs"][0].get("fully_defined"):
            break
        time.sleep(10)
    else:
        raise SystemExit(f"model {model_id} not downloaded after {timeout}s")
    try:
        es.ml.start_trained_model_deployment(model_id=model_id, wait_for="started", number_of_allocations=allocations, threads_per_allocation=1)
    except Exception as e:  # already running
        if "already" not in str(e).lower():
            raise
    print(f"model {model_id} deployed")


def ensure_index(es, index: str, model_id: str, field: str, field_type: str, recreate: bool):
    pipeline_id = f"{index}_elser_pipeline"
    if recreate and es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.ingest.put_pipeline(id=pipeline_id, body={
        "description": f"ELSER expansion of `text` into `{field}`",
        "processors": [{"inference": {"model_id": model_id, "input_output": [{"input_field": "text", "output_field": field}]}}],
    })
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0,
                         "index": {"default_pipeline": pipeline_id, "refresh_interval": "-1"}},
            "mappings": {"properties": {"doc_id": {"type": "keyword"}, "text": {"type": "text"}, field: {"type": field_type}}},
        })
        print(f"created index {index}")


def index_corpus(es, index: str, path: str, with_title: bool, chunk: int):
    from elasticsearch import helpers

    existing = es.count(index=index)["count"]
    passages = []
    for row in iter_jsonl(path):
        pid = str(row.get("_id") or row.get("id"))
        text = row.get("text", "") or ""
        if with_title and row.get("title"):
            text = f"{row['title']} {text}"
        passages.append((pid, text))
    if existing >= len(passages):
        print(f"{index}: already complete ({existing:,} docs)")
        return
    print(f"{index}: {existing:,}/{len(passages):,} indexed, resuming")

    def actions():
        for pid, text in passages:
            # `create` skips passages that are already indexed, so re-runs resume cheaply.
            yield {"_op_type": "create", "_index": index, "_id": pid, "_source": {"doc_id": pid, "text": text}}

    ok = failed = 0
    for success, info in helpers.streaming_bulk(es, actions(), chunk_size=chunk, raise_on_error=False,
                                                 max_retries=3, request_timeout=300):
        if success:
            ok += 1
        elif "version_conflict" not in str(info):
            failed += 1
    es.indices.put_settings(index=index, body={"index": {"refresh_interval": "1s"}})
    es.indices.refresh(index=index)
    print(f"{index}: +{ok:,} new, {failed:,} failed, total {es.count(index=index)['count']:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/task_a.yaml")
    ap.add_argument("--corpora", nargs="+", help="subset of corpora (default: all in the config)")
    ap.add_argument("--deploy-model", action="store_true")
    ap.add_argument("--allocations", type=int, default=1)
    ap.add_argument("--field-type", default="sparse_vector", choices=["sparse_vector", "rank_features"])
    ap.add_argument("--with-title", action="store_true", help="index `title + text` (reference notebooks: text only)")
    ap.add_argument("--recreate", action="store_true", help="delete and rebuild the index")
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()

    load_env(ROOT / ".env")
    c = cfg(a.config)
    r = c["retrieval"]
    es = ElserRetriever._connect(300)
    if a.deploy_model:
        deploy_model(es, r.get("model_id", ".elser_model_1"), a.allocations)
    for corpus in a.corpora or list(r["index_map"]):
        index = r["index_map"][corpus]
        ensure_index(es, index, r.get("model_id", ".elser_model_1"), r.get("field", "text_embedding"), a.field_type, a.recreate)
        index_corpus(es, index, str(ROOT / c["corpora"][corpus]) if not str(c["corpora"][corpus]).startswith("/") else c["corpora"][corpus],
                     a.with_title, a.chunk)


if __name__ == "__main__":
    main()
