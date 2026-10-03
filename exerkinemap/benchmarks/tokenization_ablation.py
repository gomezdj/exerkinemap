"""Matched tokenization and one-hot supervised baseline evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from exerkinemap.tokenization import (
    BpeTokenizer,
    CharacterTokenizer,
    KMerTokenizer,
    UnigramTokenizer,
    WordPieceTokenizer,
)

LOGGER = logging.getLogger(__name__)


def _input_requirement(sequence_column: str, label_column: str) -> str:
    return (
        "EXERKINEMAP does not ship labeled benchmark data. Supply a CSV derived "
        "from an empirical sequence-to-phenotype or regulatory task with "
        f"'{sequence_column}' and binary '{label_column}' columns."
    )


@dataclass(frozen=True)
class SequenceExample:
    """A binary labeled nucleotide sequence."""

    sequence: str
    label: int


@dataclass(frozen=True)
class AblationConfig:
    """Configuration shared by every ablation arm."""

    seed: int = 17
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    epochs: int = 300
    learning_rate: float = 0.1
    l2: float = 1e-4
    vocab_size: int = 256
    max_sequence_length: int = 512
    long_sequence_policy: str = "prefix"
    max_tokenizer_training_characters: int = 25_000


def _stratified_split(
    examples: Sequence[SequenceExample], config: AblationConfig
) -> tuple[list[SequenceExample], list[SequenceExample], list[SequenceExample]]:
    if not 0 < config.test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if not 0 < config.validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    by_label: dict[int, list[SequenceExample]] = {0: [], 1: []}
    for example in examples:
        if example.label not in by_label:
            raise ValueError("Only binary labels 0 and 1 are supported")
        by_label[example.label].append(example)
    label_counts = {label: len(group) for label, group in by_label.items()}
    if any(count < 3 for count in label_counts.values()):
        raise ValueError(
            "A stratified train/validation/test benchmark requires at least "
            f"three empirical examples per label; found {label_counts}."
        )

    generator = np.random.default_rng(config.seed)
    partitions: list[list[SequenceExample]] = [[], [], []]
    for group in by_label.values():
        shuffled = [group[index] for index in generator.permutation(len(group))]
        test_count = max(1, round(len(group) * config.test_fraction))
        validation_count = max(1, round(len(group) * config.validation_fraction))
        if test_count + validation_count >= len(group):
            raise ValueError("Split fractions leave no training examples for at least one label")
        partitions[2].extend(shuffled[:test_count])
        partitions[1].extend(shuffled[test_count : test_count + validation_count])
        partitions[0].extend(shuffled[test_count + validation_count :])
    return tuple(partitions)  # type: ignore[return-value]


def _make_tokenizer_factories(config: AblationConfig) -> dict[str, Callable[[], object]]:
    return {
        "character": CharacterTokenizer,
        "codon_3mer": lambda: KMerTokenizer(k=3, stride=3),
        "overlapping_6mer": lambda: KMerTokenizer(k=6, stride=1),
        "bpe": lambda: BpeTokenizer(
            max_training_characters=config.max_tokenizer_training_characters
        ),
        "unigram": lambda: UnigramTokenizer(
            max_training_characters=config.max_tokenizer_training_characters
        ),
        "wordpiece": lambda: WordPieceTokenizer(
            max_training_characters=config.max_tokenizer_training_characters
        ),
    }


def _fit_tokenizer(tokenizer: object, sequences: Iterable[str], vocab_size: int) -> object:
    fit = getattr(tokenizer, "fit", None)
    if fit is not None:
        fit(sequences, vocab_size=vocab_size)
    return tokenizer


def _window_sequence(sequence: str, config: AblationConfig) -> str:
    if config.max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    if len(sequence) <= config.max_sequence_length:
        return sequence
    if config.long_sequence_policy == "prefix":
        return sequence[: config.max_sequence_length]
    if config.long_sequence_policy == "suffix":
        return sequence[-config.max_sequence_length :]
    if config.long_sequence_policy == "center":
        start = (len(sequence) - config.max_sequence_length) // 2
        return sequence[start : start + config.max_sequence_length]
    if config.long_sequence_policy == "error":
        raise ValueError(
            "An input sequence exceeds max_sequence_length. Choose a documented "
            "long_sequence_policy or increase --max-sequence-length."
        )
    raise ValueError(
        "long_sequence_policy must be one of: prefix, suffix, center, or error"
    )


def _window_examples(
    examples: Sequence[SequenceExample], config: AblationConfig
) -> tuple[list[SequenceExample], dict[str, int | str]]:
    original_lengths = [len(example.sequence) for example in examples]
    windowed = [
        SequenceExample(_window_sequence(example.sequence, config), example.label)
        for example in examples
    ]
    return windowed, {
        "max_sequence_length": config.max_sequence_length,
        "long_sequence_policy": config.long_sequence_policy,
        "input_count": len(examples),
        "truncated_count": sum(
            length > config.max_sequence_length for length in original_lengths
        ),
        "original_max_length": max(original_lengths, default=0),
    }


def _count_vectorize(
    token_lists: Sequence[Sequence[str]], vocabulary: dict[str, int]
) -> np.ndarray:
    matrix = np.zeros((len(token_lists), len(vocabulary)), dtype=np.float64)
    for row, tokens in enumerate(token_lists):
        for token in tokens:
            column = vocabulary.get(token)
            if column is not None:
                matrix[row, column] += 1.0
        if tokens:
            matrix[row] /= len(tokens)
    return matrix


def _token_features(
    tokenizer: object,
    train_sequences: Sequence[str],
    sequences: Sequence[str],
    vocab_size: int,
) -> tuple[np.ndarray, int]:
    tokenizer = _fit_tokenizer(tokenizer, train_sequences, vocab_size)
    tokenize = getattr(tokenizer, "tokenize")
    train_tokens = [tokenize(sequence) for sequence in train_sequences]
    vocabulary = {
        token: index
        for index, token in enumerate(sorted({token for tokens in train_tokens for token in tokens}))
    }
    return _count_vectorize([tokenize(sequence) for sequence in sequences], vocabulary), len(vocabulary)


def _one_hot_features(
    train_sequences: Sequence[str], sequences: Sequence[str]
) -> tuple[np.ndarray, int]:
    alphabet = sorted({base for sequence in train_sequences for base in sequence})
    if not alphabet:
        raise ValueError("Training sequences cannot be empty")
    max_length = max(map(len, train_sequences))
    base_index = {base: index for index, base in enumerate(alphabet)}
    features = np.zeros((len(sequences), max_length * len(alphabet)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        for position, base in enumerate(sequence[:max_length]):
            index = base_index.get(base)
            if index is not None:
                features[row, position * len(alphabet) + index] = 1.0
    return features, features.shape[1]


def _standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    return tuple((matrix - mean) / scale for matrix in (train, *others))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _fit_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config: AblationConfig,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    intercept = 0.0
    best_weights = weights.copy()
    best_intercept = intercept
    best_loss = float("inf")
    for _ in range(config.epochs):
        probabilities = _sigmoid(train_x @ weights + intercept)
        residual = probabilities - train_y
        weights -= config.learning_rate * (
            (train_x.T @ residual) / len(train_y) + config.l2 * weights
        )
        intercept -= config.learning_rate * residual.mean()
        validation_probability = _sigmoid(validation_x @ weights + intercept)
        validation_loss = -np.mean(
            validation_y * np.log(validation_probability + 1e-12)
            + (1 - validation_y) * np.log(1 - validation_probability + 1e-12)
        )
        if validation_loss < best_loss:
            best_loss = float(validation_loss)
            best_weights = weights.copy()
            best_intercept = float(intercept)
    return best_weights, best_intercept


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end + 1) / 2.0
        index = end
    return ranks


def _auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both labels in the evaluation split")
    positive_ranks = _average_ranks(probabilities)[labels == 1].sum()
    return float((positive_ranks - positives * (positives + 1) / 2) / (positives * negatives))


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="mergesort")
    ordered_labels = labels[order]
    positives = int(ordered_labels.sum())
    if not positives:
        raise ValueError("Average precision requires positive labels")
    precision = np.cumsum(ordered_labels) / np.arange(1, len(ordered_labels) + 1)
    return float(precision[ordered_labels == 1].sum() / positives)


def _evaluate(
    method: str,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
    feature_count: int,
    config: AblationConfig,
) -> dict[str, float | int | str]:
    train_x, validation_x, test_x = _standardize(train_x, validation_x, test_x)
    started = time.perf_counter()
    weights, intercept = _fit_logistic(
        train_x, train_y, validation_x, validation_y, config
    )
    probabilities = _sigmoid(test_x @ weights + intercept)
    return {
        "method": method,
        "feature_count": feature_count,
        "accuracy": float(((probabilities >= 0.5) == test_y).mean()),
        "auroc": _auroc(test_y, probabilities),
        "auprc": _average_precision(test_y, probabilities),
        "brier_score": float(np.mean((probabilities - test_y) ** 2)),
        "fit_seconds": time.perf_counter() - started,
    }


def run_tokenization_ablation(
    examples: Sequence[SequenceExample], config: AblationConfig = AblationConfig()
) -> list[dict[str, float | int | str]]:
    """Evaluate learned tokenizers and a matched one-hot supervised baseline."""
    if not examples:
        raise ValueError("At least one example is required")
    windowed_examples, _ = _window_examples(examples, config)
    return _run_tokenization_ablation(windowed_examples, config)


def _run_tokenization_ablation(
    examples: Sequence[SequenceExample], config: AblationConfig
) -> list[dict[str, float | int | str]]:
    train, validation, test = _stratified_split(examples, config)
    train_sequences = [example.sequence for example in train]
    all_splits = [
        [example.sequence for example in partition]
        for partition in (train, validation, test)
    ]
    labels = [
        np.asarray([example.label for example in partition], dtype=np.float64)
        for partition in (train, validation, test)
    ]

    train_x, feature_count = _one_hot_features(train_sequences, all_splits[0])
    validation_x, _ = _one_hot_features(train_sequences, all_splits[1])
    test_x, _ = _one_hot_features(train_sequences, all_splits[2])
    results = [
        _evaluate(
            "one_hot_supervised",
            train_x,
            validation_x,
            test_x,
            *labels,
            feature_count,
            config,
        )
    ]

    for name, factory in _make_tokenizer_factories(config).items():
        tokenizer = factory()
        matrices: list[np.ndarray] = []
        vocabulary_size = 0
        for sequences in all_splits:
            features, vocabulary_size = _token_features(
                tokenizer, train_sequences, sequences, config.vocab_size
            )
            matrices.append(features)
        results.append(
            _evaluate(name, *matrices, *labels, vocabulary_size, config)
        )
    return results


def _read_examples(
    input_path: Path, sequence_column: str, label_column: str
) -> list[SequenceExample]:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Benchmark input was not found: {input_path.resolve()}. "
            f"{_input_requirement(sequence_column, label_column)}"
        )
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty. {_input_requirement(sequence_column, label_column)}")
        missing = {sequence_column, label_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {sorted(missing)}. "
                f"{_input_requirement(sequence_column, label_column)}"
            )
        examples = []
        for line_number, row in enumerate(reader, start=2):
            sequence = (row[sequence_column] or "").strip().upper()
            try:
                label = int(row[label_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid binary label at CSV line {line_number}") from error
            if sequence:
                examples.append(SequenceExample(sequence=sequence, label=label))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched tokenization and one-hot sequence-classification baselines."
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV containing labeled sequences.")
    parser.add_argument("--output", type=Path, required=True, help="CSV for benchmark results.")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seed", type=int, default=AblationConfig.seed)
    parser.add_argument("--epochs", type=int, default=AblationConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=AblationConfig.learning_rate)
    parser.add_argument("--l2", type=float, default=AblationConfig.l2)
    parser.add_argument("--vocab-size", type=int, default=AblationConfig.vocab_size)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=AblationConfig.max_sequence_length,
        help="Fixed sequence-window length applied before every benchmark arm.",
    )
    parser.add_argument(
        "--long-sequence-policy",
        choices=("prefix", "suffix", "center", "error"),
        default=AblationConfig.long_sequence_policy,
        help="How to select a window from sequences longer than the fixed length.",
    )
    parser.add_argument(
        "--max-tokenizer-training-characters",
        type=int,
        default=AblationConfig.max_tokenizer_training_characters,
        help="Maximum training characters for each learned tokenizer vocabulary.",
    )
    args = parser.parse_args()
    config = AblationConfig(
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        vocab_size=args.vocab_size,
        max_sequence_length=args.max_sequence_length,
        long_sequence_policy=args.long_sequence_policy,
        max_tokenizer_training_characters=args.max_tokenizer_training_characters,
    )
    examples = _read_examples(args.input, args.sequence_column, args.label_column)
    windowed_examples, input_summary = _window_examples(examples, config)
    label_counts = {
        str(label): sum(example.label == label for example in windowed_examples)
        for label in (0, 1)
    }
    LOGGER.info(
        "Read %d examples (%s); windowed %d of %d sequences at %d nt using %s.",
        len(windowed_examples),
        label_counts,
        input_summary["truncated_count"],
        len(windowed_examples),
        config.max_sequence_length,
        config.long_sequence_policy,
    )
    results = _run_tokenization_ablation(windowed_examples, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "input_summary": input_summary,
                "label_counts": label_counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote %d benchmark rows to %s", len(results), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
