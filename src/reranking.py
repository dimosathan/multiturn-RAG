from typing import List


def simple_rerank(passages: List[dict]):
    """
    Placeholder reranker.
    Replace with Cohere / BGE / Azure later.
    """
    return sorted(passages, key=lambda x: x.get("score", 0), reverse=True)


def normalize_scores(passages):
    if not passages:
        return passages

    scores = [p["score"] for p in passages]
    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        return passages

    for p in passages:
        p["score"] = (p["score"] - min_s) / (max_s - min_s)

    return passages
