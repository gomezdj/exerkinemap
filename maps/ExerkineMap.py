# exerkinemap/maps/ExerkineMap.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExerkineMap(nn.Module):
    """
    EXERKINEMAP: shared latent space for RNA and Protein embeddings.

    - Takes sequence-level embeddings from ExerkineRNA and ExerkineProtein.
    - Projects them into a common space.
    - Can be used with contrastive / alignment losses.
    """

    def __init__(
        self,
        in_dim_rna: int = 512,
        in_dim_protein: int = 512,
        map_dim: int = 512,
    ):
        super().__init__()

        # projection from RNA embedding → shared space
        self.rna_proj = nn.Sequential(
            nn.Linear(in_dim_rna, map_dim),
            nn.GELU(),
            nn.LayerNorm(map_dim),
        )

        # projection from Protein embedding → shared space
        self.protein_proj = nn.Sequential(
            nn.Linear(in_dim_protein, map_dim),
            nn.GELU(),
            nn.LayerNorm(map_dim),
        )

    def forward(
        self,
        rna_seq_embedding: torch.Tensor,
        protein_seq_embedding: torch.Tensor,
    ):
        """
        rna_seq_embedding: (batch, in_dim_rna)
        protein_seq_embedding: (batch, in_dim_protein)

        Returns:
            zr: (batch, map_dim) RNA in shared space
            zp: (batch, map_dim) Protein in shared space
        """
        zr = self.rna_proj(rna_seq_embedding)
        zp = self.protein_proj(protein_seq_embedding)
        return zr, zp


def contrastive_loss(
    zr: torch.Tensor,
    zp: torch.Tensor,
    temperature: float = 0.1,
):
    """
    Simple symmetric contrastive loss for RNA ↔ Protein alignment.

    zr: (batch, dim)
    zp: (batch, dim)

    Returns:
        scalar loss
    """
    zr = F.normalize(zr, dim=-1)
    zp = F.normalize(zp, dim=-1)

    logits = zr @ zp.T / temperature  # (batch, batch)
    labels = torch.arange(zr.size(0), device=zr.device)

    loss_rna_to_protein = F.cross_entropy(logits, labels)
    loss_protein_to_rna = F.cross_entropy(logits.T, labels)

    return (loss_rna_to_protein + loss_protein_to_rna) / 2.0


def build_exerkine_map(
    in_dim_rna: int = 512,
    in_dim_protein: int = 512,
    map_dim: int = 512,
) -> ExerkineMap:
    return ExerkineMap(in_dim_rna=in_dim_rna, in_dim_protein=in_dim_protein, map_dim=map_dim)
