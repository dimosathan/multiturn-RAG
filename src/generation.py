from src.answerability import classify_answerability


def extract_spans(contexts):
    """
    Placeholder for span extraction.
    """
    return contexts[:3]


def generate_candidate(contexts):
    return "Generated answer based on context."


def judge(a, b):
    """
    Placeholder judge.
    """
    return a


def run_generation_pipeline(item, config):
    contexts = item.get("contexts", [])

    label = classify_answerability(item)

    if label == "UNANSWERABLE":
        return "I don't know."

    spans = extract_spans(contexts)

    if config.get("use_dual_generation", True):
        a = generate_candidate(spans)
        b = generate_candidate(spans)
        final = judge(a, b)
    else:
        final = generate_candidate(spans)

    return final
