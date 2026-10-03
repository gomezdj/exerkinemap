from __future__ import annotations

import pytest

from exerkinemap.benchmarks.tokenization_ablation import (
    AblationConfig,
    SequenceExample,
    _read_examples,
    _window_examples,
    run_tokenization_ablation,
)
from exerkinemap.tokenization import (
    BpeTokenizer,
    KMerTokenizer,
    UnigramTokenizer,
    WordPieceTokenizer,
)


def test_codon_tokenizer_uses_non_overlapping_triplets() -> None:
    assert KMerTokenizer(k=3, stride=3).tokenize("ATGAAATTT") == ["ATG", "AAA", "TTT"]


def test_learned_tokenizers_segment_contiguous_sequences() -> None:
    sequences = ["ATGATGATG", "ATGCCCATG"]
    for tokenizer in (BpeTokenizer(), UnigramTokenizer(), WordPieceTokenizer()):
        tokenizer.fit(sequences, vocab_size=12)
        assert "".join(tokenizer.tokenize("ATGATG")) == "ATGATG"


def test_ablation_reports_all_tokenizers_and_one_hot_baseline() -> None:
    examples = [
        SequenceExample(sequence=f"ATGAAA{i % 2}CGT", label=1)
        for i in range(12)
    ] + [
        SequenceExample(sequence=f"CCCTTT{i % 2}GGA", label=0)
        for i in range(12)
    ]
    results = run_tokenization_ablation(
        examples,
        AblationConfig(epochs=30, vocab_size=16, learning_rate=0.2),
    )
    methods = {result["method"] for result in results}
    assert methods == {
        "one_hot_supervised",
        "character",
        "codon_3mer",
        "overlapping_6mer",
        "bpe",
        "unigram",
        "wordpiece",
    }
    for result in results:
        assert 0.0 <= result["auroc"] <= 1.0
        assert 0.0 <= result["auprc"] <= 1.0


def test_missing_input_explains_empirical_data_requirement(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not ship labeled benchmark data"):
        _read_examples(tmp_path / "sequence_labels.csv", "sequence", "label")


def test_empty_input_explains_empirical_data_requirement(tmp_path) -> None:
    input_path = tmp_path / "sequence_labels.csv"
    input_path.touch()
    with pytest.raises(ValueError, match="does not ship labeled benchmark data"):
        _read_examples(input_path, "sequence", "label")


def test_small_label_groups_report_observed_counts() -> None:
    examples = [
        SequenceExample(sequence="ATG", label=1),
        SequenceExample(sequence="TGA", label=0),
    ]
    with pytest.raises(ValueError, match=r"found \{0: 1, 1: 1\}"):
        run_tokenization_ablation(examples)


def test_sequence_windowing_is_deterministic_and_reported() -> None:
    examples = [SequenceExample(sequence="ATGCGTAA", label=1)]
    windowed, summary = _window_examples(
        examples,
        AblationConfig(max_sequence_length=4, long_sequence_policy="center"),
    )

    assert windowed == [SequenceExample(sequence="GCGT", label=1)]
    assert summary["truncated_count"] == 1
    assert summary["original_max_length"] == 8


def test_sequence_windowing_can_reject_long_sequences() -> None:
    with pytest.raises(ValueError, match="exceeds max_sequence_length"):
        run_tokenization_ablation(
            [
                SequenceExample(sequence="ATGCGTAA", label=1),
                SequenceExample(sequence="CCCAAATT", label=0),
                SequenceExample(sequence="ATGCGTAA", label=1),
                SequenceExample(sequence="CCCAAATT", label=0),
                SequenceExample(sequence="ATGCGTAA", label=1),
                SequenceExample(sequence="CCCAAATT", label=0),
            ],
            AblationConfig(max_sequence_length=4, long_sequence_policy="error"),
        )
