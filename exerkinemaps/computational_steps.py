# 1. Initial input
H0 = torch.cat([gene_expr, spatial_coords], dim=1)

# 2. Graph Attention layers
H1 = GATConv(input_dim, hidden_dim)(H0, edge_index)
H2 = GATConv(hidden_dim, hidden_dim)(H1, edge_index)

# 3. Optional Transformer block (long-range interactions)
# H3 = TransformerEncoder(H2)

# 4. MLP decoder
out = MLP(H2)
