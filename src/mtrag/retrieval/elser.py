"""ELSER v1 sparse retrieval over Elasticsearch (paper §4, Task A configuration).

The passage-level corpora (512 tokens, stride 100) are expected to be indexed
with an ``text_expansion`` field produced by the ``.elser_model_1`` inference
pipeline.  Index / field names are configurable (``configs/task_a.yaml``).

Environment: ``ELASTICSEARCH_CLOUD_ID`` + ``ELASTICSEARCH_PASSWORD`` (Elastic
Cloud, user ``elastic``) or ``ELASTICSEARCH_URL`` (+ optional
``ELASTICSEARCH_API_KEY``) for a self-hosted cluster.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


class ElserRetriever:
    def __init__(
        self,
        index_map: Dict[str, str],
        field: str = "text_embedding",
        model_id: str = ".elser_model_1",
        doc_id_field: str = "doc_id",
        client=None,
        request_timeout: int = 180,
    ):
        self.index_map, self.field, self.model_id, self.doc_id_field = index_map, field, model_id, doc_id_field
        self.es = client or self._connect(request_timeout)

    @staticmethod
    def _connect(timeout: int):
        from elasticsearch import Elasticsearch

        cloud_id = os.environ.get("ELASTICSEARCH_CLOUD_ID")
        if cloud_id:
            return Elasticsearch(
                cloud_id=cloud_id,
                basic_auth=(os.environ.get("ELASTICSEARCH_USER", "elastic"), os.environ["ELASTICSEARCH_PASSWORD"]),
                request_timeout=timeout,
            )
        url = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
        api_key: Optional[str] = os.environ.get("ELASTICSEARCH_API_KEY")
        return Elasticsearch(url, api_key=api_key, request_timeout=timeout) if api_key else Elasticsearch(url, request_timeout=timeout)

    def search(self, query: str, corpus: str, size: int = 100) -> List[dict]:
        """Return ``[{"document_id", "score"}, ...]`` (empty list on error)."""
        index = self.index_map.get(corpus)
        if not index or not query:
            return []
        body = {"query": {"text_expansion": {self.field: {"model_id": self.model_id, "model_text": query}}}, "size": size}
        try:
            res = self.es.search(index=index, body=body)
        except Exception:
            return []
        return [
            {"document_id": h["_source"].get(self.doc_id_field, h["_id"]), "score": h["_score"]}
            for h in res["hits"]["hits"]
        ]
