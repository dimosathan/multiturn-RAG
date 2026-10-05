"""Task A: multi-strategy rewriting, ELSER retrieval, hybrid reranking, nested RRF."""

from .fusion import NESTED_RRF_PARAMS, nested_rrf, nested_rrf_run, weighted_rrf
from .rewriters import STRATEGIES, build_rewriter

__all__ = ["NESTED_RRF_PARAMS", "STRATEGIES", "build_rewriter", "nested_rrf", "nested_rrf_run", "weighted_rrf"]
