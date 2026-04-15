import re


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_strip_quotes(text: str) -> str:
    if not text:
        return text
    return text.strip().strip('"').strip("'")


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def deduplicate_list(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
