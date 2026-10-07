"""Continue masked-language-model training from a protein FASTA corpus."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")


def read_fasta(path: Path, max_records: int | None = None) -> list[str]:
    """Read protein sequences from plain or gzipped FASTA."""
    opener = gzip.open if path.suffix == ".gz" else open
    sequences: list[str] = []
    parts: list[str] = []

    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    sequence = "".join(parts).upper()
                    if sequence and set(sequence) <= AMINO_ACIDS:
                        sequences.append(sequence)
                        if max_records is not None and len(sequences) >= max_records:
                            return sequences
                parts = []
            else:
                parts.append(line)

    if parts:
        sequence = "".join(parts).upper()
        if sequence and set(sequence) <= AMINO_ACIDS:
            sequences.append(sequence)
    return sequences


class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences: list[str], tokenizer, max_length: int):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.tokenizer(
            self.sequences[index],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )


class MaskedProteinCollator:
    """Pad a batch and apply the standard 80/10/10 MLM corruption rule."""
    def __init__(self, tokenizer, probability: float = 0.15):
        self.tokenizer = tokenizer
        self.probability = probability

    def __call__(self, records: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(records, padding=True, return_tensors="pt")
        inputs = batch["input_ids"]
        labels = inputs.clone()
        eligible = batch["attention_mask"].bool()

        for row, token_ids in enumerate(inputs.tolist()):
            special = self.tokenizer.get_special_tokens_mask(
                token_ids, already_has_special_tokens=True
            )
            eligible[row] &= torch.tensor(special, dtype=torch.bool).logical_not()

        masked = torch.bernoulli(
            torch.full(inputs.shape, self.probability, dtype=torch.float32)
        ).bool() & eligible
        # Ensure every sequence contributes at least one prediction target.
        for row in range(masked.shape[0]):
            if not masked[row].any():
                positions = torch.where(eligible[row])[0]
                if len(positions):
                    masked[row, positions[torch.randint(len(positions), (1,)).item()]] = True

        labels[~masked] = -100
        replace_with_mask = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked
        inputs[replace_with_mask] = self.tokenizer.mask_token_id
        replace_with_random = (
            torch.bernoulli(torch.full(inputs.shape, 0.5)).bool()
            & masked
            & ~replace_with_mask
        )
        random_tokens = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[replace_with_random] = random_tokens[replace_with_random]
        batch["labels"] = labels
        return batch


class OmicsPretrainer:
    """Continue pre-training an ESM-2 masked language model on protein FASTA."""

    def __init__(self, model_checkpoint: str = DEFAULT_MODEL):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    @staticmethod
    def _device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def execute_pretraining(
        self,
        dataset_path: str | Path,
        output_dir: str | Path,
        *,
        epochs: int = 1,
        batch_size: int = 4,
        max_length: int = 512,
        max_records: int | None = None,
        learning_rate: float = 5e-5,
        seed: int = 17,
        overwrite: bool = False,
    ) -> Path:
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Protein FASTA input not found: {dataset_path}")
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Choose another path or pass --overwrite."
            )
        if epochs < 1 or batch_size < 1 or max_length < 8:
            raise ValueError("epochs/batch_size must be positive and max_length must be at least 8")

        random.seed(seed)
        torch.manual_seed(seed)
        sequences = read_fasta(dataset_path, max_records=max_records)
        if not sequences:
            raise ValueError(f"No valid protein sequences found in {dataset_path}")

        dataset = ProteinSequenceDataset(sequences, self.tokenizer, max_length)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=MaskedProteinCollator(self.tokenizer),
        )
        device = self._device()
        self.model.to(device)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        epoch_losses: list[float] = []

        print(f"Training {len(dataset):,} protein sequences on {device} with {self.model_checkpoint}")
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = self.model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                steps += 1
                if steps % 100 == 0:
                    print(f"epoch {epoch + 1}/{epochs}, step {steps}/{len(loader)}, loss {total_loss / steps:.4f}")
            epoch_losses.append(total_loss / max(steps, 1))
            print(f"epoch {epoch + 1}/{epochs} complete, mean loss {epoch_losses[-1]:.4f}")

        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)
        with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "base_checkpoint": self.model_checkpoint,
                    "input_fasta": str(dataset_path),
                    "sequence_count": len(dataset),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "learning_rate": learning_rate,
                    "device": str(device),
                    "mean_epoch_losses": epoch_losses,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        return output_dir
