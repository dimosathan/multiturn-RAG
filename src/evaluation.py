def recall_at_k(preds, qrels, k=5):
    score = 0
    total = len(preds)

    for qid, docs in preds.items():
        relevant = set(qrels.get(qid, {}).keys())
        retrieved = set([d["doc_id"] for d in docs[:k]])

        if relevant & retrieved:
            score += 1

    return score / total if total > 0 else 0


def ndcg_at_k(preds, qrels, k=5):
    import math

    def dcg(scores):
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores))

    total = 0
    for qid, docs in preds.items():
        gains = []
        for d in docs[:k]:
            gains.append(qrels.get(qid, {}).get(d["doc_id"], 0))

        ideal = sorted(gains, reverse=True)

        total += dcg(gains) / (dcg(ideal) + 1e-8)

    return total / len(preds)
