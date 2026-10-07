"""
workflows/03_finetune_motrpac_fm.py

Executes supervised fine-tuning utilizing PEFT (LoRA). Training targets are 
strictly anchored to validated MoTrPAC empirical data to ensure generated 
sequences reflect highly resolved physiological exercise adaptations.
"""

from exerkinemap.foundation_model.finetuning import MoTrPACFineTuner
from peft import get_peft_model, LoraConfig, TaskType

def run_motrpac_finetuning():
    print("Loading pre-trained Omics FM...")
    finetuner = MoTrPACFineTuner(pretrained_model_path="results/pretrained_fm")
    
    # Inject LoRA configuration for Parameter-Efficient Fine-Tuning
    print("Applying LoRA adapters for compute-efficient tuning...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    finetuner.model = get_peft_model(finetuner.model, peft_config)
    finetuner.model.print_trainable_parameters()
    
    print("Initiating MoTrPAC-anchored fine-tuning...")
    output_directory = finetuner.execute_finetuning(
        motrpac_csv_path="data/raw/metadata/motrpac_targets.csv", 
        output_dir="results/finetuned_motrpac_fm"
    )
    
    print(f"MoTrPAC fine-tuning complete. Model saved to {output_directory}")

if __name__ == "__main__":
    run_motrpac_finetuning()
