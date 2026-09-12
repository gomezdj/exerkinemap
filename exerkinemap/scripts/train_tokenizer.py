"""
train_tokenizer.py

Train a BPE tokenizer for the EXERKINEMAP Genomic Language Model (GLM).
This script reads the processed RNA transcript reference file and trains a
BPE tokenizer. It uses curated-tokenizers for BPE training and gensim for
CBOW embedding initialization on the tokenized corpus.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from curated_tokenizers import Tokenizer, models, pre_tokenizers, trainers
from gensim.models import Word2Vec
from transformers import PreTrainedTokenizerFast

# Configure logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SEQ_DIR = PROJECT_ROOT / "data" / "processed" / "sequences" / "rna"
MODEL_DIR = PROJECT_ROOT / "models"
TOKENIZER_DIR = MODEL_DIR / "tokenizers"
GLM_DIR = MODEL_DIR / "glm"


def create_directories():
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    GLM_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Verified tokenizer output directory: {TOKENIZER_DIR}")
    logger.info(f"Verified GLM output directory: {GLM_DIR}")


def load_transcript_sequences(reference_path: Path):
    if not reference_path.exists():
        logger.error(
            f"RNA transcript reference not found: {reference_path}."
            " Run workflows/04_build_sequence_reference.py first."
        )
        raise FileNotFoundError(reference_path)

    logger.info(f"Loading transcript reference from {reference_path}")
    df = pd.read_parquet(reference_path, columns=["sequence"])
    if "sequence" not in df.columns:
        raise ValueError("Transcript reference file must contain a 'sequence' column.")

    return df["sequence"].astype(str).tolist()


def train_bpe_tokenizer(sequences, vocab_size=10000):
    logger.info("Training BPE tokenizer on RNA transcript sequences...")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    tokenizer.train_from_iterator(sequences, trainer=trainer)

    tokenizer_path = TOKENIZER_DIR / "glm_bpe_tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    logger.info(f"Saved tokenizer to {tokenizer_path}")

    return tokenizer_path


def build_tokenized_corpus(tokenizer_path: Path, sequences):
    logger.info("Loading trained tokenizer and preparing tokenized corpus...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )

    tokenized = []
    for seq in sequences:
        tokens = tokenizer.tokenize(seq)
        if tokens:
            tokenized.append(tokens)
    logger.info(f"Tokenized {len(tokenized)} sequences for CBOW training.")
    return tokenized


def train_cbow_embeddings(tokenized_corpus, vector_size=256, window=2):
    logger.info("Training gensim CBOW embeddings on the tokenized corpus...")
    cbow_model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=0,
        workers=os.cpu_count() or 4,
    )

    cbow_path = GLM_DIR / "cbow_embeddings.model"
    cbow_model.save(str(cbow_path))
    logger.info(f"Saved CBOW embeddings to {cbow_path}")
    return cbow_path


def main():
    parser = argparse.ArgumentParser(description="Train the EXERKINEMAP GLM tokenizer.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Vocabulary size for the BPE tokenizer.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=256,
        help="Vector size for gensim CBOW embeddings.",
    )
    args = parser.parse_args()

    create_directories()
    reference_path = PROCESSED_SEQ_DIR / "transcript_reference.parquet"
    sequences = load_transcript_sequences(reference_path)

    tokenizer_path = train_bpe_tokenizer(sequences, vocab_size=args.vocab_size)
    tokenized_corpus = build_tokenized_corpus(tokenizer_path, sequences)
    cbow_path = train_cbow_embeddings(tokenized_corpus, vector_size=args.vector_size)

    logger.info("Tokenizer and CBOW embedding training complete.")
    logger.info(f"Saved tokenizer: {tokenizer_path}")
    logger.info(f"Saved CBOW model: {cbow_path}")


if __name__ == "__main__":
    main()
