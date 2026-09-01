# exerkinemap/src/run_exerkinemap.py

import torch
import torch.nn as nn
import torch.optim as optim

from exerkinemap.models.ExerkineRNA import ExerkineRNA, rna_seq_to_ids
from exerkinemap.models.ExerkineProtein import (
    ExerkineProtein,
    protein_seq_to_ids,
    protein_features_to_ids,
)
from exerkinemap.maps.ExerkineMap import (
    ExerkineMap,
    contrastive_loss,
)

# -----------------------------
# Device (macOS MPS-friendly)
# -----------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------------
# Build models
# -----------------------------
rna_model = ExerkineRNA().to(DEVICE)
protein_model = ExerkineProtein().to(DEVICE)
map_model = ExerkineMap().to(DEVICE)

# -----------------------------
# Optimizer
# -----------------------------
optimizer = optim.AdamW(
    list(rna_model.parameters())
    + list(protein_model.parameters())
    + list(map_model.parameters()),
    lr=1e-4,
)

# -----------------------------
# Example batch (replace with loader)
# -----------------------------
rna_seqs = [
    "AUGCUUAGCUAGCUAGC",
    "GGGAAAUUCCCGGAUUA",
]

protein_seqs = [
    "MKTLLILAVVAFVLSA",
    "MGSSHHHHHHSSGLVPR",
]

# -----------------------------
# Tokenize RNA
# -----------------------------
rna_ids_batch = [rna_seq_to_ids(seq) for seq in rna_seqs]
max_rna_len = max(len(x) for x in rna_ids_batch)
rna_tensor = torch.zeros(len(rna_ids_batch), max_rna_len, dtype=torch.long)

for i, ids in enumerate(rna_ids_batch):
    rna_tensor[i, : len(ids)] = torch.tensor(ids)

rna_tensor = rna_tensor.to(DEVICE)

# -----------------------------
# Tokenize Protein + Features
# -----------------------------
protein_ids_batch = [protein_seq_to_ids(seq) for seq in protein_seqs]
protein_feat_batch = [protein_features_to_ids(seq) for seq in protein_seqs]

max_pro_len = max(len(x) for x in protein_ids_batch)
protein_tensor = torch.zeros(len(protein_ids_batch), max_pro_len, dtype=torch.long)

for i, ids in enumerate(protein_ids_batch):
    protein_tensor[i, : len(ids)] = torch.tensor(ids)

protein_tensor = protein_tensor.to(DEVICE)

# -----------------------------
# Forward pass
# -----------------------------
rna_out = rna_model(rna_tensor)
protein_out = protein_model(protein_tensor, protein_feat_batch)

rna_seq_embed = rna_out["seq_embedding"]          # (batch, dim)
protein_seq_embed = protein_out["seq_embedding"]  # (batch, dim)

# -----------------------------
# Map into shared latent space
# -----------------------------
zr, zp = map_model(rna_seq_embed, protein_seq_embed)

# -----------------------------
# Compute contrastive loss
# -----------------------------
loss = contrastive_loss(zr, zp)
print(f"Contrastive loss: {loss.item():.4f}")

# -----------------------------
# Backprop + update
# -----------------------------
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Training step complete.")
