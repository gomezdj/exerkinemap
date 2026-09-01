import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- device: macOS MPS ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- simple RNA tokenizer ----
VOCAB = ["A", "C", "G", "U", "N", "[PAD]", "[MASK]"]
token2id = {t: i for i, t in enumerate(VOCAB)}
pad_id = token2id["[PAD]"]
mask_id = token2id["[MASK]"]
vocab_size = len(VOCAB)

# ---- simple state-space block (toy SSM) ----
class SimpleSSMBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear_in = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.linear_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        residual = x
        x = self.linear_in(x)
        x = x.transpose(1, 2)          # (batch, d_model, seq)
        x = self.conv(x)
        x = x.transpose(1, 2)          # (batch, seq, d_model)
        x = self.linear_out(x)
        x = self.norm(x + residual)
        return x

# ---- attention block ----
class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        residual = x
        x_attn, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x_attn + residual)
        residual = x
        x_ff = self.ff(x)
        x = self.norm2(x_ff + residual)
        return x

# ---- hybrid 12-layer RNA-LM ----
class ExerkineRNALM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(10000, d_model)  # max length
        blocks = []
        for i in range(n_layers):
            if i % 2 == 0:
                blocks.append(SimpleSSMBlock(d_model))   # even: SSM
            else:
                blocks.append(AttentionBlock(d_model, n_heads))  # odd: attention
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # input_ids: (batch, seq)
        bsz, seqlen = input_ids.size()
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.embed(input_ids) + self.pos_embed(pos)

        # no causal mask for bidirectional MLM; add if you want autoregressive
        for block in self.blocks:
            if isinstance(block, AttentionBlock):
                x = block(x, attn_mask=None)
            else:
                x = block(x)

        x = self.norm(x)
        logits = self.head(x)  # (batch, seq, vocab)
        return logits

# ---- instantiate on MPS ----
model = ExerkineRNALM(vocab_size=vocab_size).to(device)

# ---- dummy batch ----
batch_tokens = torch.randint(0, vocab_size, (2, 512), device=device)
logits = model(batch_tokens)
print(logits.shape)  # (2, 512, vocab_size)
