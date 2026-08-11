"""
05_train_glm.py

This script trains the Genomic Language Model (GLM) for the EXERKINEMAP framework.
Following the mathematical model, it executes:
1. BPE/k-mer Tokenization.
2. CBOW Genomic Embeddings to minimize L_CBOW.
3. BERT-Style Language Model initialized with H^(0) = E_CBOW + P.
"""
import os
import torch
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

# Configure logging
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SEQ_DIR = PROJECT_ROOT / "data" / "processed" / "sequences" / "rna"
MODEL_DIR = PROJECT_ROOT / "models"
GLM_DIR = MODEL_DIR / "glm"
TOKENIZER_DIR = MODEL_DIR / "tokenizers"

def create_directories():
    """Ensure output model directories exist."""
    GLM_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

def train_bpe_tokenizer(sequences, vocab_size=10000):
    """
    BPE Tokenization
    Merges the most frequent neighboring nucleotide pairs iteratively.
    """
    logger.info("Training BPE Tokenizer on nucleotide sequences...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    # Train directly on the in-memory sequence list
    tokenizer.train_from_iterator(sequences, trainer=trainer)
    
    tokenizer_path = TOKENIZER_DIR / "glm_bpe_tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]"
    )

def train_cbow_embeddings(tokenized_corpus, vocab_size, vector_size=256, window=2):
    """
    Section 7: CBOW Genomic Embeddings
    Optimizes L_CBOW = sum_i log P(t_i | C_i) for contextual initialization.
    """
    logger.info(f"Training CBOW model with window={window}, vector_size={vector_size}...")
    
    # gensim Word2Vec with sg=0 computes CBOW
    cbow_model = Word2Vec(
        sentences=tokenized_corpus, 
        vector_size=vector_size, 
        window=window, 
        min_count=1, 
        sg=0,
        workers=4
    )
    
    cbow_path = GLM_DIR / "cbow_embeddings.model"
    cbow_model.save(str(cbow_path))
    logger.info(f"CBOW embeddings saved to {cbow_path}")
    
    return cbow_model

def initialize_bert_with_cbow(cbow_model, tokenizer, vector_size=256, max_position_embeddings=512):
    """
    Section 8: BERT-Style Genomic Language Model
    H^(0) = E_CBOW + P. Injects the CBOW weights into the BERT Embedding layer.
    """
    logger.info("Initializing BERT configuration...")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=vector_size,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    model = BertForMaskedLM(config)
    
    logger.info("Injecting CBOW weights into BERT token embeddings (E_CBOW)...")
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data
    
    # Map gensim CBOW weights to the PyTorch embedding layer
    for word, idx in tokenizer.get_vocab().items():
        if word in cbow_model.wv:
            embedding_matrix[idx] = torch.tensor(cbow_model.wv[word])
            
    return model

def main():
    logger.info("Initializing 05_train_glm workflow...")
    create_directories()

    # 1. Load Data
    ref_path = PROCESSED_SEQ_DIR / "transcript_reference.parquet"
    if not ref_path.exists():
        logger.error(f"Reference not found at {ref_path}. Run 04_build_sequence_reference.py first.")
        return
        
    logger.info("Loading RNA transcript references...")
    df = pd.read_parquet(ref_path)
    
    # For memory and time efficiency during initial setup, take a subset if necessary
    sequences = df['sequence'].tolist()
    
    # 2. Train Tokenizer (Section 6.1)
    tokenizer = train_bpe_tokenizer(sequences, vocab_size=10000)
    
    # 3. Prepare corpus for CBOW
    logger.info("Tokenizing corpus for CBOW...")
    tokenized_corpus = [tokenizer.tokenize(seq) for seq in sequences]
    
    # 4. Train CBOW Embeddings (Section 7)
    cbow_model = train_cbow_embeddings(tokenized_corpus, vocab_size=tokenizer.vocab_size, vector_size=256)
    
    # 5. Initialize BERT Model (Section 8)
    model = initialize_bert_with_cbow(cbow_model, tokenizer, vector_size=256)
    
    # 6. Prepare HuggingFace Dataset for Trainer
    logger.info("Preparing dataset for Masked Language Modeling (MLM)...")
    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"], 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )
    
    hf_dataset = Dataset.from_pandas(df[['sequence']])
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["sequence"])
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )
    
    # 7. Train GLM
    training_args = TrainingArguments(
        output_dir=str(GLM_DIR / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=500,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    logger.info("Starting GLM Training (Transformer layers)...")
    trainer.train()
    
    # Save final model
    final_model_path = GLM_DIR / "final_model"
    trainer.save_model(str(final_model_path))
    logger.info(f"GLM training complete. Model saved to {final_model_path}")

if __name__ == "__main__":
    main()
