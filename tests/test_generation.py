"""Task B pipeline: prompt fidelity against the original implementation + unit tests."""

import json

import pytest
from conftest import load_fixture

from mtrag.generation import ExtractivenessBand, GenerationConfig, GenerationPipeline, Models, selection_scores
from mtrag.generation.text import detect_question_type, extractiveness, is_pure_idk, remove_forbidden
from mtrag.llm import FakeChatModel

GOLD = load_fixture("golden_generation.json")


def responder(system, user):
    if system.startswith("Extract"):
        return json.dumps({"extractedSpans": [
            {"passageId": 1, "sentence": "The Eiffel Tower is 330 metres tall and located in Paris France."},
            {"passageId": 2, "sentence": "It was completed in 1889 for the World Fair."}]})
    if system.startswith("Helpful, accurate"):
        return "The Eiffel Tower is 330 metres tall and located in Paris France. It was completed in 1889."
    if system == "Judge assistant.":
        return '{"winner":"A","score_A":8,"score_B":6,"reason":"x"}'
    if system == "User perspective.":
        return '{"preferred":"B","confidence":"MEDIUM"}'
    if system == "Editor.":
        return ("The Eiffel Tower, located in Paris France, is 330 metres tall. It was completed in 1889 for the World Fair, "
                "and it remains one of the most visited monuments in the world today, attracting millions of visitors each "
                "and every year from many countries.")
    return "The information is not available."


def make_pipeline(**cfg):
    fake = FakeChatModel(responder)
    return GenerationPipeline(Models(fake, fake, fake, fake, fake), GenerationConfig(**cfg)), fake


def test_prompts_and_output_identical_to_original():
    # user_judge_rate=1.0 because the golden run forced the user judge.
    pipe, fake = make_pipeline(user_judge_rate=1.0)
    row = pipe.process_task(GOLD["task"])
    got = [(c["system"], c["user"], c["temperature"], c["max_tokens"]) for c in fake.calls]
    exp = [(c["system"], c["user"], c["temperature"], c["max_tokens"]) for c in GOLD["calls"]]
    assert got == exp
    assert row["predictions"][0]["text"] == GOLD["output"]


def test_no_context_refusal_identical():
    pipe, fake = make_pipeline()
    row = pipe.process_task(GOLD["task2"])
    assert [(c["system"], c["user"]) for c in fake.calls] == [(c["system"], c["user"]) for c in GOLD["calls2"]]
    assert row["predictions"][0]["text"] == GOLD["output2"] and row["contexts"] == []


def test_phi_band():
    b = ExtractivenessBand()
    assert [b.phi(x) for x in (0.1, 0.28, 0.38, 0.45, 0.5, 0.9)] == [-1.5, 2.5, 2.5, 1.5, 1.5, 0.5]


def test_selection_score_equation():
    spans = [{"text": "alpha beta gamma delta epsilon zeta"}]
    a, b = "alpha beta gamma delta epsilon zeta", "i don't know omega"
    s = selection_scores(a, b, spans, {"score_A": 6, "score_B": 10}, None)
    assert s["A"] == pytest.approx(0.35 * 6 + 2.0 + 0.5)          # r4 = 1.0 -> over-extractive
    assert s["B"] == pytest.approx(0.35 * 10 - 1.5 - 2.0)          # r4 = 0, forbidden phrase
    s2 = selection_scores(a, b, spans, {}, {"preferred": "B", "confidence": "HIGH"})
    assert s2["B"] == pytest.approx(0.35 * 5 + 5.0 - 1.5 - 2.0)


def test_text_helpers():
    assert extractiveness("a b c d", [{"text": "x a b c d y"}]) == 1.0
    assert extractiveness("a b c", [{"text": "a b c"}]) == 0.0  # fewer than 4 tokens
    assert is_pure_idk("I don't know.") and not is_pure_idk("I don't know " + "word " * 40)
    assert remove_forbidden("It is big, I'm not sure.") == "It is big,."  # faithful to the original cleanup
    assert detect_question_type("How do I reset it?") == "how_to"
    assert detect_question_type("pricing tiers") == "keyword"
    assert detect_question_type("Who wrote it?") == "factoid"


def test_user_judge_sampling_is_deterministic_and_calibrated():
    pipe, _ = make_pipeline(user_judge_rate=0.6)
    draws = [pipe.sample_user_judge(f"task{i}") for i in range(4000)]
    assert draws == [pipe.sample_user_judge(f"task{i}") for i in range(4000)]
    assert 0.57 < sum(draws) / len(draws) < 0.63
