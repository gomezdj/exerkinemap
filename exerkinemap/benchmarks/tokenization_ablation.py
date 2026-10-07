"""
exerkinemap/benchmarks/tokenization_ablation.py

Executes a matched binary sequence-classification benchmark across diverse tokenization 
strategies. Trains vocabularies exclusively on the training partition and evaluates 
using a fixed linear-model optimization budget (Logistic Regression) to compare 
sequence parsing efficacy (Accuracy, AUROC, AUPRC).
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.feature_extraction.text import CountVectorizer
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

def encode_one_hot(sequences, max_len=None):
    """Position-aware one-hot supervised control."""
    chars = sorted(list(set("".join(sequences))))
    char_to_int = {c: i for i, c in enumerate(chars)}
    
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
        
    encoded = np.zeros((len(sequences), max_len * len(chars)), dtype=np.int8)
    for i, seq in enumerate(sequences):
        for j, char in enumerate(seq[:max_len]):
            if char in char_to_int:
                encoded[i, j * len(chars) + char_to_int[char]] = 1
    return encoded

def train_hf_tokenizer(model_type, sequences, vocab_size=1000):
    """Trains a HuggingFace tokenizer (BPE, Unigram, or WordPiece) on the training set."""
    if model_type == "BPE":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    elif model_type == "Unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    elif model_type == "WordPiece":
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    else:
        raise ValueError("Unsupported tokenizer type")

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(sequences, trainer=trainer)
    return tokenizer

def extract_features(train_seqs, test_seqs, strategy, max_len=None):
    """Extracts features based on the selected tokenization strategy."""
    if strategy == "one-hot":
        X_train = encode_one_hot(train_seqs, max_len)
        X_test = encode_one_hot(test_seqs, X_train.shape[1] // len(set("".join(train_seqs))))
        return X_train, X_test

    if strategy in ["char", "3-mer", "6-mer"]:
        # Non-overlapping for codon (3-mer), overlapping for 6-mer, character for char
        if strategy == "char":
            analyzer, ngram = 'char', (1, 1)
        elif strategy == "3-mer":
            # Hack to simulate non-overlapping codons with CountVectorizer: insert spaces
            train_seqs = [" ".join([s[i:i+3] for i in range(0, len(s), 3)]) for s in train_seqs]
            test_seqs = [" ".join([s[i:i+3] for i in range(0, len(s), 3)]) for s in test_seqs]
            analyzer, ngram = 'word', (1, 1)
        else:
            analyzer, ngram = 'char', (6, 6)
            
        vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=1000)
        X_train = vectorizer.fit_transform(train_seqs)
        X_test = vectorizer.transform(test_seqs)
        return X_train, X_test

    if strategy in ["BPE", "Unigram", "WordPiece"]:
        tokenizer = train_hf_tokenizer(strategy, train_seqs)
        
        # Tokenize and stringify for CountVectorizer logic (bag-of-subwords)
        train_tokens = [" ".join(tokenizer.encode(seq).tokens) for seq in train_seqs]
        test_tokens = [" ".join(tokenizer.encode(seq).tokens) for seq in test_seqs]
        
        vectorizer = CountVectorizer(analyzer='word')
        X_train = vectorizer.fit_transform(train_tokens)
        X_test = vectorizer.transform(test_tokens)
        return X_train, X_test

def main():
    parser = argparse.ArgumentParser(description="Tokenization Ablation and One-Hot Control")
    parser.add_argument("--input", type=str, required=True, help="Path to empirical CSV with `sequence` and `label` columns.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path for benchmarking results.")
    args = parser.parse_args()

    print(f"Loading empirical sequences from {args.input}...")
    df = pd.read_csv(args.input)
    
    if 'sequence' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input CSV must contain 'sequence' and 'label' columns.")

    # Stratified Train/Test Split
    X = df['sequence'].tolist()
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    strategies = ["one-hot", "char", "3-mer", "6-mer", "BPE", "Unigram", "WordPiece"]
    results = []

    print("Executing tokenization ablation benchmarking...")
    for strategy in strategies:
        print(f"  -> Training and evaluating {strategy}...")
        
        # 1. Feature Extraction (Train vocabularies strictly on X_train)
        X_train_feat, X_test_feat = extract_features(X_train, X_test, strategy)
        feature_count = X_train_feat.shape[1]

        # 2. Linear-Model Optimization Budget
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        clf.fit(X_train_feat, y_train)
        
        # 3. Evaluation
        y_pred = clf.predict(X_test_feat)
        y_prob = clf.predict_proba(X_test_feat)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)

        results.append({
            "Method": strategy,
            "Accuracy": acc,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Feature_Count": feature_count
        })

    # Save Results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"Benchmarking complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()