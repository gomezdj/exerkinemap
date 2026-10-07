"""
exerkinemap/foundation_model/embeddings.py

Handles the tokenization and latent space integration for the EXERKINEMAP 
three-stage Omics Foundation Model (FM) pipeline. Converts raw nucleotide 
(ExerkineRNA) and amino-acid (ExerkineProtein) sequences into contextualized 
algebraic embeddings mapped via positional encoders.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class OmicsEmbedder(nn.Module):
    def __init__(self, 
                 protein_model_checkpoint: str = "facebook/esm2_t33_650M_UR50D",
                 rna_model_checkpoint: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
                 unified_dim: int = 768):
        """
        Initializes the genomic (gLM) and proteomic (pLM) tokenizers and base models.
        Includes a projection layer to map disparate embedding dimensions into a 
        unified latent space for spatial ligand-receptor communication mapping.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Optimizes for M3 Pro Apple Silicon
        else:
            self.device = torch.device("cpu")

        # ExerkineProtein (PLM) Initialization
        self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_checkpoint)
        self.protein_model = AutoModel.from_pretrained(protein_model_checkpoint).to(self.device)
        
        # ExerkineProtein (PLM) Initialization
        self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_checkpoint)
        self.protein_model = AutoModel.from_pretrained(protein_model_checkpoint).to(self.device)
        self.protein_dim = self.protein_model.config.hidden_size
        
        # ExerkineRNA (GLM) Initialization
        self.rna_tokenizer = AutoTokenizer.from_pretrained(rna_model_checkpoint)
        self.rna_model = AutoModel.from_pretrained(rna_model_checkpoint).to(self.device)
        self.rna_dim = self.rna_model.config.hidden_size

        # Unified Latent Space Projections
        self.protein_projection = nn.Linear(self.protein_dim, unified_dim).to(self.device)
        self.rna_projection = nn.Linear(self.rna_dim, unified_dim).to(self.device)

    def generate_protein_embeddings(self, sequences: list) -> torch.Tensor:
        """
        Tokenizes and generates amino-acid embeddings using ExerkineProtein (e.g., ESM-2).
        Returns pooled representations projected to the unified latent dimension.
        """
        inputs = self.protein_tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.protein_model(**inputs)
            # Use mean pooling across the sequence length for a sequence-level embedding
            pooled_output = outputs.last_hidden_state.mean(dim=1)
            
        return self.protein_projection(pooled_output)

    def generate_rna_embeddings(self, sequences: list) -> torch.Tensor:
        """
        Tokenizes and generates nucleotide embeddings using ExerkineRNA (gLM).
        Returns pooled representations projected to the unified latent dimension.
        """
        inputs = self.rna_tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.rna_model(**inputs)
            # Use mean pooling for a sequence-level genomic embedding
            pooled_output = outputs.last_hidden_state.mean(dim=1)
            
        return self.rna_projection(pooled_output)

    def build_multimodal_batch(self, protein_seqs: list, rna_seqs: list) -> dict:
        """
        Constructs a unified dictionary of embeddings ready for causal-set 
        propagation and fine-tuning against MoTrPAC physiological targets.
        """
        protein_embeds = self.generate_protein_embeddings(protein_seqs) if protein_seqs else None
        rna_embeds = self.generate_rna_embeddings(rna_seqs) if rna_seqs else None
        
        return {
            "exerkine_protein_embeddings": protein_embeds,
            "exerkine_rna_embeddings": rna_embeds
        }

if __name__ == "__main__":
    # Example execution for standardizing embeddings prior to MoTrPAC fine-tuning
    embedder = OmicsEmbedder()
    
    sample_proteins = ["MKVLLILACLVALALA", "MGFVLRRDWR"]
    sample_rnas = ["ATGCGTACGTAGCTAG", "CGCGCATATATCGCG"]
    
    unified_batch = embedder.build_multimodal_batch(sample_proteins, sample_rnas)
    
    if unified_batch["exerkine_protein_embeddings"] is not None:
        print(f"Protein latent space shape: {unified_batch['exerkine_protein_embeddings'].shape}")
    if unified_batch["exerkine_rna_embeddings"] is not None:
        print(f"RNA latent space shape: {unified_batch['exerkine_rna_embeddings'].shape}")