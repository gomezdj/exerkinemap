"""
06_generate_protein_embeddings.py

This script implements the Protein Language Model (PLM) for the EXERKINEMAP framework.
Following the mathematical model (Sections 5.2 and 9), it utilizes a generative PLM 
(ProGen2 architecture) to:
1. Tokenize amino acid sequences: t^P_q = T_P(a^P_q)
2. Extract contextual embeddings: z^P_q = f_{theta_P}(t^P_q)
3. Support autoregressive generation: P_PLM(a^P) = prod P(a_t | a_{<t})
"""
import os
import torch
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SEQ_DIR = PROJECT_ROOT / "data" / "processed" / "sequences" / "protein"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
MODEL_DIR = PROJECT_ROOT / "models" / "plm"

# PLM Settings (ProGen2 is explicitly referenced in the architecture)
PLM_MODEL_NAME = "Salesforce/progen2-large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_directories():
    """Ensure output model and embedding directories exist."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_plm():
    """
    Load the Tokenizer T_P and the Protein Language Model f_{theta_P}.
    """
    logger.info(f"Loading PLM ({PLM_MODEL_NAME}) onto {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    
    # We use a Causal LM to satisfy both the embedding extraction 
    # and the P_PLM(a^P) autoregressive generation requirements.
    model = AutoModelForCausalLM.from_pretrained(PLM_MODEL_NAME, output_hidden_states=True)
    model.to(DEVICE)
    model.eval()
    
    return tokenizer, model

def compute_protein_embeddings(df, tokenizer, model, batch_size=8, max_length=512):
    """
    Section 5.2: Extract z^P_q = f_{theta_P}(t^P_q).
    Processes sequences in batches, applying mean-pooling to the last hidden state.
    """
    logger.info("Computing z^P_q embeddings for protein sequences...")
    all_embeddings = []
    
    sequences = df['sequence'].tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]
            
            # t^P_q = T_P(a^P_q)
            inputs = tokenizer(
                batch_seqs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_length
            ).to(DEVICE)
            
            # f_{theta_P}(t^P_q)
            outputs = model(**inputs)
            
            # Extract the last hidden states
            hidden_states = outputs.hidden_states[-1] 
            
            # Create attention mask for proper mean pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            
            # z^P_q (mean pooled representation)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            all_embeddings.extend(batch_embeddings)
            
    return np.array(all_embeddings)

def generate_candidate_sequence(tokenizer, model, prompt_sequence, max_new_tokens=50):
    """
    Section 9: Candidate Sequence Generation.
    Samples a^P_hat ~ P_PLM(a^P | z_q) using autoregressive generation.
    """
    logger.info(f"Generating candidate extension for prompt: {prompt_sequence}")
    
    inputs = tokenizer(prompt_sequence, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    candidate_sequence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return candidate_sequence

def main():
    logger.info("Initializing 06_generate_protein_embeddings workflow...")
    create_directories()

    # 1. Load Data
    ref_path = PROCESSED_SEQ_DIR / "protein_reference.parquet"
    if not ref_path.exists():
        logger.error(f"Reference not found at {ref_path}. Run 04_build_sequence_reference.py first.")
        return
        
    logger.info("Loading Protein references...")
    df = pd.read_parquet(ref_path)
    
    # 2. Initialize Model (T_P and f_{theta_P})
    tokenizer, model = load_plm()
    
    # 3. Compute Embeddings (z^P_q)
    # Using a subset for demonstration/testing speed if dataset is massive
    # df = df.head(1000) 
    embeddings = compute_protein_embeddings(df, tokenizer, model)
    
    # 4. Save Embeddings
    output_path = EMBEDDINGS_DIR / "plm_embeddings.npy"
    np.save(output_path, embeddings)
    logger.info(f"Successfully saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} to {output_path}")

    # 5. Optional: Demonstrate Generation (a^P_hat ~ P_PLM)
    # Example starting motif for an exerkine (e.g., a signal peptide sequence)
    sample_prompt = "MKWVTFISLL" 
    candidate = generate_candidate_sequence(tokenizer, model, sample_prompt)
    logger.info(f"Generated Candidate Sequence: {candidate}")
    
    logger.info("Workflow 06_generate_protein_embeddings complete.")

if __name__ == "__main__":
    main()
