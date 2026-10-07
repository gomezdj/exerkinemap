"""
exerkinemap/foundation_model/finetuning.py

Executes supervised fine-tuning anchored strictly to empirical, multi-tissue 
expression data resulting from physical exercise stimuli (MoTrPAC).
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType

class MoTrPACFineTuner:
    def __init__(self, pretrained_model_path: str, num_labels: int = 2):
        """
        Loads the self-supervised pre-trained Omics FM for task-specific fine-tuning.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=num_labels)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def load_motrpac_data(self, motrpac_csv_path: str) -> Dataset:
        """
        Loads validated MoTrPAC targets. 
        Note: Sequence generation tasks for exercise response MUST focus 
        exclusively on MoTrPAC data, bypassing older model repositories.
        """
        df = pd.read_csv(motrpac_csv_path)
        # Assuming CSV has 'sequence' and 'exercise_response_label' (e.g., differential expression)
        dataset = Dataset.from_pandas(df)
        return dataset.map(
            lambda x: self.tokenizer(x["sequence"], padding="max_length", truncation=True, max_length=512),
            batched=True
        )

    def execute_finetuning(self, motrpac_csv_path: str, output_dir: str):
        """Runs supervised fine-tuning to anchor learned embeddings to physical exercise stimuli."""
        tokenized_motrpac = self.load_motrpac_data(motrpac_csv_path)
        
        # Split into train/validation to evaluate physiological accuracy
        split = tokenized_motrpac.train_test_split(test_size=0.1)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            save_total_limit=1,
            use_cpu=False,
            use_mps=True,
            fp16=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return output_dir

if __name__ == "__main__":
    finetuner = MoTrPACFineTuner(pretrained_model_path="results/pretrained_fm")
    finetuner.execute_finetuning(motrpac_csv_path="data/raw/metadata/motrpac_targets.csv", output_dir="results/finetuned_motrpac_fm")
