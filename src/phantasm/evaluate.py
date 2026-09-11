"""Persona fidelity evaluation.

The question this module answers is narrow and measurable: *do the model's
replies look like the target person's replies?* It compares generated responses
against the held-out ground truth using interpretable, deterministic, local
metrics. No metric here captures "personality"; each one measures a specific,
named surface property of how somebody writes.

Every metric is a similarity in ``[0, 1]`` computed from two corpora of
responses. The Persona Fidelity Score is the weighted mean of the component
metrics multiplied by 100 (see :data:`METRIC_WEIGHTS`), and the components are
always reported alongside it.
"""

import json
import math
import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from phantasm.dataset import read_sharegpt, split_sample
from phantasm.storage import write_json_atomic
from phantasm.text import (
    PUNCTUATION_MARKS,
    count_emoji,
    sentences,
    strip_code_blocks,
    word_tokens,
)

#: Component metrics and their weight in the aggregate score.
METRIC_WEIGHTS: dict[str, float] = {
    "length_similarity": 1.0,
    "punctuation_similarity": 1.0,
    "emoji_style_similarity": 1.0,
    "casing_similarity": 1.0,
    "lexical_similarity": 1.0,
    "phrase_similarity": 1.0,
    "sentence_similarity": 1.0,
}

#: How many of the most frequent terms enter the lexical and phrase vectors.
VOCABULARY_LIMIT = 400

ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def ratio_similarity(first: float, second: float) -> float:
    """Similarity of two non-negative magnitudes: ``min/max``, 1.0 when both are 0."""
    if first < 0 or second < 0:
        raise ValueError("ratio_similarity needs non-negative values")
    if not first and not second:
        return 1.0
    return min(first, second) / max(first, second)


def rate_similarity(first: float, second: float) -> float:
    """Similarity of two rates already expressed in ``[0, 1]``."""
    return max(0.0, 1.0 - abs(first - second))


def cosine(first: Counter, second: Counter) -> float:
    """Cosine similarity of two frequency vectors; 1.0 when both are empty."""
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    shared = set(first) & set(second)
    numerator = sum(first[key] * second[key] for key in shared)
    norm = math.sqrt(sum(v * v for v in first.values())) * math.sqrt(
        sum(v * v for v in second.values())
    )
    return numerator / norm if norm else 0.0


def ngrams(tokens: list[str], size: int) -> Iterable[tuple[str, ...]]:
    return (tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def style_profile(texts: list[str]) -> dict[str, Any]:
    """Reduce a corpus of responses to the surface features the metrics compare."""
    texts = [text for text in texts if isinstance(text, str) and text.strip()]
    if not texts:
        raise ValueError("A style profile needs at least one non-empty response")
    words = [word_tokens(text) for text in texts]
    word_counts = [len(w) for w in words]
    characters = [len(text) for text in texts]
    all_tokens = [token for group in words for token in group]
    punctuation = Counter(
        character for text in texts for character in text if character in PUNCTUATION_MARKS
    )
    total_characters = sum(characters) or 1
    letters = [c for text in texts for c in text if c.isalpha()]
    sentence_lengths = [
        len(word_tokens(sentence)) for text in texts for sentence in sentences(text)
    ]
    unigrams = Counter(all_tokens)
    phrases = Counter()
    for group in words:
        phrases.update(ngrams(group, 2))
        phrases.update(ngrams(group, 3))
    return {
        "responses": len(texts),
        "mean_words": statistics.fmean(word_counts),
        "median_words": statistics.median(word_counts),
        "mean_characters": statistics.fmean(characters),
        "punctuation": Counter(
            {key: value / total_characters for key, value in punctuation.items()}
        ),
        "punctuation_rate": sum(punctuation.values()) / total_characters,
        "emoji_per_response": sum(count_emoji(text) for text in texts) / len(texts),
        "emoji_response_share": sum(1 for text in texts if count_emoji(text)) / len(texts),
        "uppercase_ratio": (sum(1 for c in letters if c.isupper()) / len(letters))
        if letters
        else 0.0,
        "lowercase_message_share": sum(
            1 for text in texts if text == text.lower() and any(c.isalpha() for c in text)
        )
        / len(texts),
        "shouting_share": sum(
            1
            for text in texts
            if any(c.isalpha() for c in text) and strip_code_blocks(text).isupper()
        )
        / len(texts),
        "type_token_ratio": (len(set(all_tokens)) / len(all_tokens)) if all_tokens else 0.0,
        "unigrams": Counter(dict(unigrams.most_common(VOCABULARY_LIMIT))),
        "phrases": Counter(dict(phrases.most_common(VOCABULARY_LIMIT))),
        "mean_sentence_words": statistics.fmean(sentence_lengths) if sentence_lengths else 0.0,
        "sentences_per_response": len(sentence_lengths) / len(texts),
    }


def compare_profiles(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    """Component metrics comparing a candidate corpus with the ground truth."""
    components = {
        "length_similarity": statistics.fmean(
            [
                ratio_similarity(reference["mean_words"], candidate["mean_words"]),
                ratio_similarity(reference["median_words"], candidate["median_words"]),
                ratio_similarity(reference["mean_characters"], candidate["mean_characters"]),
            ]
        ),
        "punctuation_similarity": statistics.fmean(
            [
                cosine(reference["punctuation"], candidate["punctuation"]),
                ratio_similarity(reference["punctuation_rate"], candidate["punctuation_rate"]),
            ]
        ),
        "emoji_style_similarity": statistics.fmean(
            [
                ratio_similarity(reference["emoji_per_response"], candidate["emoji_per_response"]),
                rate_similarity(
                    reference["emoji_response_share"], candidate["emoji_response_share"]
                ),
            ]
        ),
        "casing_similarity": statistics.fmean(
            [
                rate_similarity(reference["uppercase_ratio"], candidate["uppercase_ratio"]),
                rate_similarity(
                    reference["lowercase_message_share"], candidate["lowercase_message_share"]
                ),
                rate_similarity(reference["shouting_share"], candidate["shouting_share"]),
            ]
        ),
        "lexical_similarity": statistics.fmean(
            [
                cosine(reference["unigrams"], candidate["unigrams"]),
                rate_similarity(reference["type_token_ratio"], candidate["type_token_ratio"]),
            ]
        ),
        "phrase_similarity": cosine(reference["phrases"], candidate["phrases"]),
        "sentence_similarity": statistics.fmean(
            [
                ratio_similarity(
                    reference["mean_sentence_words"], candidate["mean_sentence_words"]
                ),
                ratio_similarity(
                    reference["sentences_per_response"], candidate["sentences_per_response"]
                ),
            ]
        ),
    }
    return {name: round(value, 4) for name, value in components.items()}


def fidelity_score(components: dict[str, float]) -> float:
    """Weighted mean of the component metrics, scaled to 0-100."""
    missing = set(METRIC_WEIGHTS) - set(components)
    if missing:
        raise ValueError(f"Missing component metrics: {', '.join(sorted(missing))}")
    total = sum(METRIC_WEIGHTS.values())
    weighted = sum(components[name] * weight for name, weight in METRIC_WEIGHTS.items())
    return round(100 * weighted / total, 1)


def to_chat_messages(context: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert ShareGPT context turns into role/content chat messages."""
    return [{"role": ROLE_MAP[turn["from"]], "content": turn["value"]} for turn in context]


def generate_responses(
    rows: list[dict],
    generator: Callable[[list[dict[str, str]]], str],
    *,
    limit: int | None = None,
) -> list[str]:
    """Ask a generator for one response per held-out sample."""
    selected = rows[:limit] if limit else rows
    outputs = []
    for row in selected:
        context, _ = split_sample(row)
        reply = generator(to_chat_messages(context))
        outputs.append(reply if isinstance(reply, str) else "")
    return outputs


def reference_responses(rows: list[dict], *, limit: int | None = None) -> list[str]:
    """The real target responses for the same samples ``generate_responses`` answers."""
    return [split_sample(row)[1] for row in (rows[:limit] if limit else rows)]


def evaluate_generations(reference: list[str], candidates: dict[str, list[str]]) -> dict[str, Any]:
    """Score one or more candidate corpora against the ground-truth responses."""
    reference_profile = style_profile(reference)
    results = {}
    for name, texts in candidates.items():
        usable = [text for text in texts if isinstance(text, str) and text.strip()]
        if not usable:
            raise ValueError(f"{name} produced no non-empty responses to score")
        components = compare_profiles(reference_profile, style_profile(usable))
        results[name] = {
            "responses": len(usable),
            "empty_responses": len(texts) - len(usable),
            "components": components,
            "persona_fidelity_score": fidelity_score(components),
        }
    return {
        "held_out_responses": reference_profile["responses"],
        "reference": {
            "mean_words": round(reference_profile["mean_words"], 2),
            "median_words": reference_profile["median_words"],
            "emoji_per_response": round(reference_profile["emoji_per_response"], 3),
            "type_token_ratio": round(reference_profile["type_token_ratio"], 3),
        },
        "models": results,
        "weights": METRIC_WEIGHTS,
    }


class LlamaCppGenerator:
    """Generate replies from a local GGUF model using its own chat template."""

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 2048,
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 3407,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        system_prompt: str = "",
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "Install local inference with: uv sync --locked --extra inference"
            ) from None
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt

    def __call__(self, messages: list[dict[str, str]]) -> str:
        if self.system_prompt and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}, *messages]
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        content = response["choices"][0]["message"]["content"]
        return content.strip() if isinstance(content, str) else ""

    def close(self) -> None:
        self.llm.close()


def load_predictions(path: str) -> list[str]:
    """Read pre-generated responses: one JSON string or ``{"response": ...}`` per line."""
    texts = []
    for number, line in enumerate(
        Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"{path}:{number}: invalid JSON") from None
        if isinstance(value, dict):
            value = value.get("response", value.get("value"))
        if not isinstance(value, str):
            raise ValueError(f"{path}:{number}: expected a string response")
        texts.append(value)
    if not texts:
        raise ValueError(f"No predictions found in {path}")
    return texts


def evaluate_dataset(
    dataset_path: str,
    generators: dict[str, Callable[[list[dict[str, str]]], str]],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate and score responses for a held-out ShareGPT dataset."""
    rows, info = read_sharegpt(dataset_path)
    reference = reference_responses(rows, limit=limit)
    candidates = {
        name: generate_responses(rows, generator, limit=limit)
        for name, generator in generators.items()
    }
    result = evaluate_generations(reference, candidates)
    result["dataset"] = {"path": info["path"], "sha256": info["sha256"], "samples": len(rows)}
    return result


def render(result: dict[str, Any]) -> str:
    """Format an evaluation for a terminal, components first."""
    lines = [
        "Phantasm Persona Evaluation",
        "─" * 46,
        "",
        f"{'Held-out responses':<30}{result['held_out_responses']}",
        "",
    ]
    names = list(result["models"])
    width = max((len(name) for name in names), default=10) + 2
    header = f"{'Metric':<30}" + "".join(f"{name:>{width}}" for name in names)
    lines += [header, "─" * len(header)]
    for metric in METRIC_WEIGHTS:
        label = metric.replace("_", " ").capitalize()
        row = f"{label:<30}"
        for name in names:
            row += f"{result['models'][name]['components'][metric]:>{width}.2f}"
        lines.append(row)
    lines.append("─" * len(header))
    score_row = f"{'Persona Fidelity Score':<30}"
    for name in names:
        score_row += f"{result['models'][name]['persona_fidelity_score']:>{width}.1f}"
    lines += [score_row, ""]
    if len(names) == 2 and "fine-tuned" in names and "base" in names:
        delta = (
            result["models"]["fine-tuned"]["persona_fidelity_score"]
            - result["models"]["base"]["persona_fidelity_score"]
        )
        lines.append(f"Fine-tuning changed the score by {delta:+.1f} points.")
    lines += [
        "Scores are style similarity to the held-out responses, not a measure of",
        "whether the model is factually or behaviourally the same person.",
    ]
    return "\n".join(lines)


def add_arguments(parser: Any) -> None:
    """Register ``phantasm evaluate`` options."""
    parser.add_argument("dataset", help="Held-out ShareGPT JSONL, normally the test split")
    parser.add_argument("-m", "--model", help="GGUF model to evaluate")
    parser.add_argument(
        "--baseline-model", help="Second GGUF (usually the un-tuned base) to compare against"
    )
    parser.add_argument(
        "--predictions", help="Score pre-generated responses (JSONL) instead of running a model"
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N held-out samples")
    parser.add_argument("-s", "--system-prompt", default="", help="System prompt for generation")
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("-k", "--max-tokens", type=int, default=150)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("-o", "--output", help="Write the JSON result to this path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")


def _generator(args: Any, model_path: str) -> LlamaCppGenerator:
    return LlamaCppGenerator(
        model_path,
        n_ctx=args.context_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
        system_prompt=args.system_prompt,
    )


def command(args: Any) -> None:
    """Score a model, or pre-generated responses, against held-out target replies."""
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not args.model and not args.predictions:
        raise ValueError("Supply --model to generate responses, or --predictions to score a file")
    if args.predictions:
        rows, info = read_sharegpt(args.dataset)
        reference = reference_responses(rows, limit=args.limit)
        predictions = load_predictions(args.predictions)
        if len(predictions) < len(reference):
            raise ValueError(
                f"{args.predictions} has {len(predictions)} responses for "
                f"{len(reference)} held-out samples"
            )
        result = evaluate_generations(reference, {"fine-tuned": predictions[: len(reference)]})
        result["dataset"] = {"path": info["path"], "sha256": info["sha256"], "samples": len(rows)}
    else:
        generators = {"fine-tuned": _generator(args, args.model)}
        if args.baseline_model:
            generators["base"] = _generator(args, args.baseline_model)
        try:
            result = evaluate_dataset(args.dataset, generators, limit=args.limit)
        finally:
            for generator in generators.values():
                generator.close()
    if args.output:
        write_json_atomic(Path(args.output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else render(result))
