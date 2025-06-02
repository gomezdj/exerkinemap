# File: src/models/lri_spatial_gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class LRISpatialGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        super().__init__()
        self.gnn1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.1)
        self.gnn2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, batch):
        h = self.gnn1(x, edge_index)
        h = self.gnn2(h, edge_index)
        h_pool = global_mean_pool(h, batch)
        return self.classifier(h_pool)
