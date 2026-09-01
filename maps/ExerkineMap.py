class ExerkineMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rna_proj = nn.Linear(dim, dim)
        self.protein_proj = nn.Linear(dim, dim)

    def forward(self, rna_embed, protein_embed):
        zr = self.rna_proj(rna_embed)
        zp = self.protein_proj(protein_embed)
        return zr, zp
