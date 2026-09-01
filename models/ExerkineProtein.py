# exerkinemap/models/ExerkineProtein.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- device: macOS MPS-friendly ----
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- amino acid vocabulary ----
AA_VOCAB = [
    "A", "C", "D", "E", "F",
    "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R",
    "S", "T", "V", "W", "Y",
    "[PAD]"
]
aa2id = {aa: i for i, aa in enumerate(AA_VOCAB)}
PAD_ID = aa2id["[PAD]"]
VOCAB_SIZE = len(AA_VOCAB)

# ---- residue feature categories ----
RESIDUE_FEATURES = {
    "A": ["nonpolar"],
    "V": ["nonpolar"],
    "L": ["nonpolar"],
    "I": ["nonpolar"],
    "M": ["nonpolar", "sulfur"],
    "F": ["aromatic", "nonpolar"],
    "W": ["aromatic", "nonpolar"],
    "Y": ["aromatic", "polar"],
    "S": ["polar"],
    "T": ["polar"],
    "N": ["polar"],
    "Q": ["polar"],
    "C": ["polar", "sulfur"],
    "G": ["special"],
    "P": ["special"],
    "H": ["positive"],
    "K": ["positive"],
    "R": ["positive"],
    "D": ["negative"],
    "E": ["negative"],
}

FEATURE_TYPES = ["nonpolar", "polar", "aromatic", "positive", "negative", "sulfur", "special"]
feature2id = {f: i for i, f in enumerate(FEATURE_TYPES)}
NUM_FEATURES = len(FEATURE_TYPES)


def protein_seq_to_ids(seq: str):
    return [aa2id.get(aa, PAD_ID) for aa in seq]


def protein_features_to_ids(seq: str):
    ids = []
    for aa in seq:
        feats = RESIDUE_FEATURES.get(aa, [])
        feat_ids = [feature2id[f] for f in feats]
        ids.append(feat_ids)
    return ids


class SimpleSSMBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear_in = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.linear_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        residual = x
        x = self.linear_in(x)
        x = x.transpose(1, 2)          # (batch, d_model, seq)
        x = self.conv(x)
        x = x.transpose(1, 2)          # (batch, seq, d_model)
        x = self.linear_out(x)
        x = self.norm(x + residual)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_attn, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x_attn + residual)
        residual = x
        x_ff = self.ff(x)
        x = self.norm2(x_ff + residual)
        return x


class ExerkineProtein(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        max_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model

        # sequence + position embeddings
        self.embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_len, d_model)

        # residue feature embedding
        self.feature_embed = nn.Embedding(NUM_FEATURES, d_model)

        # hybrid backbone: even = SSM, odd = attention
        blocks = []
        for i in range(n_layers):
            if i % 2 == 0:
                blocks.append(SimpleSSMBlock(d_model))
            else:
                blocks.append(AttentionBlock(d_model, n_heads))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(d_model)

        # heads: MLM + sequence-level projection
        self.mlm_head = nn.Linear(d_model, VOCAB_SIZE)
        self.seq_head = nn.Linear(d_model, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        feature_ids_batch: list[list[list[int]]] | None = None,
        attn_mask: torch.Tensor | None = None,
    ):
        """
        input_ids: (batch, seq) integer AA IDs
        feature_ids_batch: list[batch][seq][feature_indices]
        """
        bsz, seqlen = input_ids.size()
        device = input_ids.device

        # positional encoding
        pos = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.embed(input_ids) + self.pos_embed(pos)  # (batch, seq, d_model)

        # residue feature embeddings
        if feature_ids_batch is not None:
            max_feats = max(len(f) for f in feature_ids_batch[0])
            feat_tensor = torch.zeros(bsz, seqlen, max_feats, dtype=torch.long, device=device)

            for b in range(bsz):
                for i, feats in enumerate(feature_ids_batch[b]):
                    for j, f in enumerate(feats):
                        feat_tensor[b, i, j] = f

            feat_embed = self.feature_embed(feat_tensor)      # (batch, seq, max_feats, d_model)
            feat_embed = feat_embed.sum(dim=2)                # (batch, seq, d_model)
            x = x + feat_embed

        # backbone
        for block in self.blocks:
            if isinstance(block, AttentionBlock):
                x = block(x, attn_mask=attn_mask)
            else:
                x = block(x)

        x = self.norm(x)

        # per-residue logits (MLM)
        mlm_logits = self.mlm_head(x)  # (batch, seq, vocab)

        # sequence-level embedding (mean pool)
        seq_repr = x.mean(dim=1)       # (batch, d_model)
        seq_out = self.seq_head(seq_repr)

        return {
            "mlm_logits": mlm_logits,
            "seq_embedding": seq_out,
        }


def build_exerkine_protein() -> ExerkineProtein:
    model = ExerkineProtein()
    return model.to(DEVICE)
