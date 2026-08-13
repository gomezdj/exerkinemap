"""Protein sequence generation scaffolding for the MoTrPAC framework."""

from __future__ import annotations
import torch
from copy import deepcopy
from typing import Union

# Transformers and Modeling
from transformers import AutoModelForCausalLM, AutoTokenizer, EsmForMaskedLM, EsmTokenizer
import esm
from tqdm import tqdm

# Structural and Biological Modules
from biotite.database.rcsb import fetch
from biotite.structure import AtomArray
from language import (
    FixedLengthSequenceSegment,
    MaximizePLDDT,
    MaximizePTM,
    MinimizeCRmsd,
    MinimizeDRmsd,
    MinimizeSurfaceHydrophobics,
    ProgramNode,
    pdb_file_to_atomarray,
    sequence_from_atomarray,
)

class ESM2Generator:
    """
    Generates and refines sequences using ESM-2 via Masked Language Modeling (MLM).
    Ideal for optimizing specific functional motifs in ligands or receptors.
    """
    def __init__(self, model_checkpoint: str = "facebook/esm2_t30_150M_UR50D"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = EsmTokenizer.from_pretrained(model_checkpoint)
        self.model = EsmForMaskedLM.from_pretrained(model_checkpoint).to(self.device)
        self.model.eval()

    def refine_sequence(self, sequence: str, mask_indices: list[int]) -> str:
        """Replaces specified indices with <mask> tokens and predicts the most likely amino acids."""
        seq_list = list(sequence)
        for idx in mask_indices:
            seq_list[idx] = self.tokenizer.mask_token
        
        masked_seq = "".join(seq_list)
        inputs = self.tokenizer(masked_seq, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        # Extract the highest probability token for each masked position
        for idx in mask_indices:
            # Offset by 1 to account for the BOS token
            token_idx = inputs.input_ids[0].tolist().index(self.tokenizer.mask_token_id)
            best_token = logits[0, token_idx].argmax(dim=-1).item()
            seq_list[idx] = self.tokenizer.decode([best_token])
            
        return "".join(seq_list)

class ProGen2Generator:
    """
    Autoregressive sequence generation using ProGen2.
    Well-suited for generating novel exerkine continuations from an N-terminal seed.
    """
    def __init__(self, model_checkpoint: str = "jinyuan22/ProGen2-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ProGen2 requires trust_remote_code=True for its custom architecture
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True).to(self.device)
        self.model.eval()

    def generate(self, seed_context: str, max_length: int = 150) -> str:
        """Generates a sequence continuation. ProGen2 often uses '1' as an N-terminal initialization token."""
        formatted_seed = f"1{seed_context}"
        inputs = self.tokenizer(formatted_seed, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.92,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
            
        # Decode and strip the initialization token
        generated_seq = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_seq.lstrip("1")

class SWATFeatureExtractor:
    """
    Extracts mean representations from generated sequences to feed into the GOME predictive algorithm.
    Mirrors the extraction logic found in the SWAT repository.
    """
    def __init__(self, model_checkpoint: str = "esm2_t30_150M_UR50D"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = self.model.num_layers

    def extract_mean_embeddings(self, sequence_data: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        """
        Takes a list of (ID, Sequence) tuples and returns a dictionary of mean embeddings.
        """
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequence_data)
        batch_tokens = batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        
        mean_representations = {}
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
            token_representations = results["representations"][self.repr_layer]
            
            for i, tokens_len in enumerate(batch_lens):
                # Exclude <bos> and <eos> tokens for the mean pooling
                sequence_repr = token_representations[i, 1 : tokens_len - 1].mean(0).cpu()
                mean_representations[batch_labels[i]] = sequence_repr
                
        return mean_representations

class ProteinGenerator:
    """Simple placeholder for protein sequence generation."""
    def generate(self, seed: str = "", length: int = 16) -> str:
        return seed or "M" * length

def process_protein_sequence(
    generator_type: str = "progen2", 
    seed: str = "M", 
    length: int = 150
) -> Union[str, dict[str, torch.Tensor]]:
    """
    High-level wrapper to dispatch generation (ProGen2, ESM-2) 
    or feature extraction (SWAT) to the chosen pLM.
    """
    if generator_type == "progen2":
        generator = ProGen2Generator()
        return generator.generate(seed_context=seed, max_length=length)
        
    elif generator_type == "esm2":
        generator = ESM2Generator()
        # Dynamically assign mask indices based on seed length to avoid out-of-bounds errors
        mask_indices = [1, 2, 3] if len(seed) > 3 else []
        return generator.refine_sequence(sequence=seed, mask_indices=mask_indices)
        
    elif generator_type == "swat":
        # SWAT extracts embeddings for downstream regression in GOME
        extractor = SWATFeatureExtractor()
        # SWAT expects a list of (ID, Sequence) tuples
        sequence_data = [("target_seed", seed)] 
        return extractor.extract_mean_embeddings(sequence_data=sequence_data)
        
    else:
        raise ValueError("Unsupported generator type. Choose 'progen2', 'esm2', or 'swat'.")

def generate_target_backbone(pdb_id: str) -> ProgramNode:
    """
    Generates a ProgramNode to constrain sequence generation 
    to the fixed backbone of a target structure.
    """
    # Fetch the template dynamically
    template_atoms: AtomArray = pdb_file_to_atomarray(fetch(pdb_id, format="pdb"))
    
    # Determine sequence length from the target's atom array
    sequence_length = len(sequence_from_atomarray(template_atoms))
    sequence = FixedLengthSequenceSegment(sequence_length)
    
    # Return the node with energy terms optimizing for the specific folding domains
    return ProgramNode(
        sequence_segment=sequence,
        energy_function_terms=[
            MaximizePTM(),
            MaximizePLDDT(),
            MinimizeSurfaceHydrophobics(),
            MinimizeCRmsd(template=template_atoms, backbone_only=True),
            MinimizeDRmsd(template=template_atoms, backbone_only=True),
        ],
    )