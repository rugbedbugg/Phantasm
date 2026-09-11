"""Persona fidelity metrics: determinism, direction and aggregation."""

import json

import pytest

from phantasm.evaluate import (
    METRIC_WEIGHTS,
    compare_profiles,
    cosine,
    evaluate_dataset,
    evaluate_generations,
    fidelity_score,
    generate_responses,
    load_predictions,
    rate_similarity,
    ratio_similarity,
    reference_responses,
    render,
    style_profile,
    to_chat_messages,
)

TARGET = [
    "yeah lol same",
    "idk man",
    "haha true",
    "nah i'm good",
    "ok cool see you then",
]
IMPOSTOR = [
    "I would be delighted to assist you with that request today.",
    "Certainly. Here is a comprehensive overview of the situation.",
    "Please let me know if there is anything further I can do.",
    "That is an excellent question, and I am happy to elaborate.",
    "In summary, the matter appears to be entirely resolved.",
]


def test_similarity_helpers_are_bounded():
    assert ratio_similarity(0, 0) == 1.0
    assert ratio_similarity(2, 4) == 0.5
    assert rate_similarity(0.2, 0.9) == pytest.approx(0.3)
    assert cosine({}, {}) == 1.0
    assert cosine({"a": 1}, {}) == 0.0
    with pytest.raises(ValueError):
        ratio_similarity(-1, 1)


def test_identical_corpora_score_perfectly():
    components = compare_profiles(style_profile(TARGET), style_profile(TARGET))
    assert set(components) == set(METRIC_WEIGHTS)
    assert all(value == 1.0 for value in components.values())
    assert fidelity_score(components) == 100.0


def test_metrics_are_deterministic_across_runs():
    first = compare_profiles(style_profile(TARGET), style_profile(IMPOSTOR))
    second = compare_profiles(style_profile(TARGET), style_profile(IMPOSTOR))
    assert first == second


def test_a_different_voice_scores_lower_than_the_real_one():
    close = fidelity_score(compare_profiles(style_profile(TARGET), style_profile(TARGET[:3])))
    far = fidelity_score(compare_profiles(style_profile(TARGET), style_profile(IMPOSTOR)))
    assert far < close


def test_individual_metrics_track_the_property_they_name():
    reference = style_profile(["hey 😀", "sure 😀", "ok 😀"])
    no_emoji = compare_profiles(reference, style_profile(["hey", "sure", "ok"]))
    with_emoji = compare_profiles(reference, style_profile(["yes 😀", "no 😀", "maybe 😀"]))
    assert with_emoji["emoji_style_similarity"] > no_emoji["emoji_style_similarity"]

    shouting = compare_profiles(reference, style_profile(["HEY", "SURE", "OK"]))
    assert shouting["casing_similarity"] < with_emoji["casing_similarity"]

    long_form = compare_profiles(reference, style_profile(["a much longer reply than usual " * 4]))
    assert long_form["length_similarity"] < with_emoji["length_similarity"]


def test_fidelity_score_is_the_documented_weighted_mean():
    components = dict.fromkeys(METRIC_WEIGHTS, 0.5)
    assert fidelity_score(components) == 50.0
    with pytest.raises(ValueError, match="Missing component metrics"):
        fidelity_score({"length_similarity": 1.0})


def test_profiles_need_content():
    with pytest.raises(ValueError, match="at least one non-empty response"):
        style_profile(["", "   "])


def test_evaluate_generations_compares_several_models():
    result = evaluate_generations(TARGET, {"fine-tuned": TARGET, "base": IMPOSTOR})
    assert result["held_out_responses"] == 5
    assert result["models"]["fine-tuned"]["persona_fidelity_score"] == 100.0
    assert (
        result["models"]["base"]["persona_fidelity_score"]
        < result["models"]["fine-tuned"]["persona_fidelity_score"]
    )
    rendered = render(result)
    assert "Persona Fidelity Score" in rendered
    assert "Lexical similarity" in rendered
    assert "Fine-tuning changed the score by" in rendered


def test_empty_generations_are_rejected():
    with pytest.raises(ValueError, match="no non-empty responses"):
        evaluate_generations(TARGET, {"fine-tuned": ["", "  "]})


def dataset(tmp_path, responses):
    path = tmp_path / "dataset_test_sharegpt.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "conversations": [
                        {"from": "system", "value": "persona"},
                        {"from": "human", "value": f"question {index}"},
                        {"from": "gpt", "value": response},
                    ]
                }
            )
            + "\n"
            for index, response in enumerate(responses)
        )
    )
    return str(path)


def test_generation_uses_context_without_the_answer(tmp_path):
    from phantasm.dataset import read_sharegpt

    rows, _ = read_sharegpt(dataset(tmp_path, TARGET))
    seen = []

    def generator(messages):
        seen.append(messages)
        return "canned reply"

    outputs = generate_responses(rows, generator, limit=2)
    assert outputs == ["canned reply", "canned reply"]
    assert [message["role"] for message in seen[0]] == ["system", "user"]
    assert all("yeah lol same" not in message["content"] for message in seen[0])
    assert reference_responses(rows, limit=2) == TARGET[:2]


def test_evaluate_dataset_scores_a_stub_generator(tmp_path):
    path = dataset(tmp_path, TARGET)
    replies = iter(TARGET)
    result = evaluate_dataset(path, {"fine-tuned": lambda messages: next(replies)})
    assert result["models"]["fine-tuned"]["persona_fidelity_score"] == 100.0
    assert result["dataset"]["samples"] == 5
    assert len(result["dataset"]["sha256"]) == 64


def test_chat_message_conversion_maps_sharegpt_roles():
    converted = to_chat_messages(
        [{"from": "system", "value": "s"}, {"from": "human", "value": "h"}]
    )
    assert converted == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "h"},
    ]


def test_predictions_files_accept_strings_and_objects(tmp_path):
    path = tmp_path / "preds.jsonl"
    path.write_text('"first"\n{"response": "second"}\n')
    assert load_predictions(str(path)) == ["first", "second"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_predictions(str(bad))


def test_unicode_and_emoji_corpora_are_scored(tmp_path):
    result = evaluate_generations(
        ["ça va 😀", "très bien"], {"fine-tuned": ["ça va 😀", "très bien"]}
    )
    assert result["models"]["fine-tuned"]["persona_fidelity_score"] == 100.0
