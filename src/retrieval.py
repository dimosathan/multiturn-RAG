from src.fusion import nested_rrf


def rewrite_all(query, config):
    """
    Placeholder for all rewriting strategies.
    Replace with your notebook logic.
    """
    rewrites = {
        "minimal": query,
        "hyde": query,
        "cot": query,
        "anchor": query,
        "corpus": query
    }
    return rewrites


def retrieve_passages(query, top_k=10):
    """
    Dummy retrieval.
    Replace with Elasticsearch (ELSER).
    """
    return [
        {"doc_id": f"doc_{i}", "score": 1.0 / (i + 1)}
        for i in range(top_k)
    ]


def run_retrieval_pipeline(item, config):
    query = item.get("query", "")

    rewrites = rewrite_all(query, config)

    rewrite_results = {}

    for name, q in rewrites.items():
        rewrite_results[name] = retrieve_passages(q, config["top_k_retrieve"])

    fused = nested_rrf(
        rewrite_results,
        weights=config.get("weights"),
        top_k=config.get("top_k_final", 10)
    )

    return fused
