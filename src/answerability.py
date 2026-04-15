def classify_answerability(item):
    """
    Placeholder.
    Replace with DeepSeek model logic.
    """
    return "ANSWERABLE"


def is_answerable(label):
    return label in ["ANSWERABLE", "PARTIAL"]


def normalize_answer(answer):
    if not answer:
        return ""

    answer = answer.strip()

    if answer.lower() in ["i don't know", "unknown"]:
        return "I don't know."

    return answer
