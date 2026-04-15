# =============================================================================
# Final configuration — fixed after dev set experiments
# All values here reflect the submitted system for SemEval 2026 Task 8
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_PATH = os.getenv("BASE_PATH", "./")

DATA_DIR        = os.path.join(BASE_PATH, "data")
CORPORA_DIR     = os.path.join(DATA_DIR, "corpora")
RETRIEVAL_DIR   = os.path.join(DATA_DIR, "retrieval")
PREDICTIONS_DIR = os.path.join(BASE_PATH, "predictions")
RESULTS_DIR     = os.path.join(BASE_PATH, "results")

EVAL_SCRIPT     = os.path.join(BASE_PATH, "run_retrieval_eval.py")

# ---------------------------------------------------------------------------
# Corpora & Elasticsearch index config
# ---------------------------------------------------------------------------
CORPORA_CFG = {
    "clapnq": {
        "corpus_file":    os.path.join(CORPORA_DIR, "clapnq.jsonl"),
        "queries_file":   os.path.join(RETRIEVAL_DIR, "clapnq", "clapnq_lastturn.jsonl"),
        "qrels_file":     os.path.join(RETRIEVAL_DIR, "clapnq", "dev.tsv"),
        "index_name":     "clapnq-elser-v1",
        "collection_id":  "mt-rag-clapnq-elser-512-100-20240503",
    },
    "fiqa": {
        "corpus_file":    os.path.join(CORPORA_DIR, "fiqa.jsonl"),
        "queries_file":   os.path.join(RETRIEVAL_DIR, "fiqa", "fiqa_lastturn.jsonl"),
        "qrels_file":     os.path.join(RETRIEVAL_DIR, "fiqa", "dev.tsv"),
        "index_name":     "fiqa-elser-v1",
        "collection_id":  "mt-rag-fiqa-beir-elser-512-100-20240501",
    },
    "govt": {
        "corpus_file":    os.path.join(CORPORA_DIR, "govt.jsonl"),
        "queries_file":   os.path.join(RETRIEVAL_DIR, "govt", "govt_lastturn.jsonl"),
        "qrels_file":     os.path.join(RETRIEVAL_DIR, "govt", "dev.tsv"),
        "index_name":     "govt-elser-v1",
        "collection_id":  "mt-rag-govt-elser-512-100-20240611",
    },
    "cloud": {
        "corpus_file":    os.path.join(CORPORA_DIR, "cloud.jsonl"),
        "queries_file":   os.path.join(RETRIEVAL_DIR, "cloud", "cloud_lastturn.jsonl"),
        "qrels_file":     os.path.join(RETRIEVAL_DIR, "cloud", "dev.tsv"),
        "index_name":     "cloud-elser-v1",
        "collection_id":  "mt-rag-ibmcloud-elser-512-100-20240502",
    },
}

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K        = 100   # ELSER retrieval size
RETRIEVAL_TOP_K_LARGE  = 1000  # used for upper-bound analysis

# Query rewriting: history window (fixed after ablations)
REWRITER_USER_TURNS      = 6
REWRITER_ASSISTANT_TURNS = 3

# ---------------------------------------------------------------------------
# Reranking — Cohere Rerank v4
# ---------------------------------------------------------------------------
COHERE_TOP_K_RERANK   = 100   # how many docs passed to Cohere
COHERE_HYBRID_WEIGHT  = 0.5   # final weight: 0.5*ELSER + 0.5*Cohere

# ---------------------------------------------------------------------------
# Nested RRF — final per-corpus configuration (locked from dev experiments)
# ---------------------------------------------------------------------------
NESTED_RRF_CONFIGS = {
    "clapnq": {
        "k_final":    20,
        "k_internal": 40,
        "weights": {
            "StructuralMinimal":   0.55,
            "CorpusSpecific":      0.40,
            "WeakGroupConsensus":  0.05,
        },
    },
    "fiqa": {
        "k_final":    60,
        "k_internal": 40,
        "weights": {
            "StructuralMinimal":   0.45,
            "CorpusSpecific":      0.40,
            "WeakGroupConsensus":  0.15,
        },
    },
    "govt": {
        "k_final":    40,
        "k_internal": 40,
        "weights": {
            "StructuralMinimal":   0.65,
            "CorpusSpecific":      0.25,
            "WeakGroupConsensus":  0.10,
        },
    },
    "cloud": {
        "k_final":    20,
        "k_internal": 40,
        "weights": {
            "StructuralMinimal":   0.65,
            "CorpusSpecific":      0.30,
            "WeakGroupConsensus":  0.05,
        },
    },
}

# Weak group consensus: internal weights for AnchorKeyword + COT + Hyde
WEAK_INTERNAL_WEIGHTS = {
    "AnchorKeyword": 0.33,
    "COT":           0.33,
    "Hyde":          0.34,
}

# ---------------------------------------------------------------------------
# LLM Reranking — DeepSeek-R1 listwise (test set only)
# ---------------------------------------------------------------------------
R1_TOP_K_RERANK  = 20     # top-k passed to DeepSeek-R1
R1_DOC_TRUNCATE  = 2500   # characters per document
R1_HYBRID_WEIGHT = 0.5    # final weight: 0.5*R1 + 0.5*RRF score
