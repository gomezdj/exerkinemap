"""BERT-style scaffolding for the GLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BertConfig:
    """Minimal configuration object for a BERT-like model."""

    vocab_size: int
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 2


class BertForMaskedLM:
    """A minimal placeholder BERT model for masked language modeling."""

    def __init__(self, config: BertConfig):
        self.config = config
        self.embeddings = np.random.randn(config.vocab_size, config.hidden_size).astype(float)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        return self.embeddings[input_ids]


def build_bert_model(vocab_size: int, hidden_size: int = 64) -> BertForMaskedLM:
    """Construct a minimal BERT-like model."""
    return BertForMaskedLM(BertConfig(vocab_size=vocab_size, hidden_size=hidden_size))
