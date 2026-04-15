from collections import defaultdict


def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc["doc_id"]
            scores[doc_id] += 1.0 / (k + rank + 1)

    return scores


def nested_rrf(rewrite_results, weights=None, k=60, top_k=10):
    """
    rewrite_results: dict[strategy_name -> list[doc dict]]
    """

    if weights is None:
        weights = {key: 1.0 for key in rewrite_results}

    all_scores = defaultdict(float)

    for strategy, docs in rewrite_results.items():
        weight = weights.get(strategy, 1.0)

        for rank, doc in enumerate(docs):
            doc_id = doc["doc_id"]
            score = 1.0 / (k + rank + 1)
            all_scores[doc_id] += weight * score

    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    return [{"doc_id": doc_id, "score": score} for doc_id, score in ranked[:top_k]]
