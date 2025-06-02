# File: src/models/spatial_lri_gnn.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import softmax

class SpatialLRIGNN(nn.Module):
    def __init__(self, input_dim, lri_feat_dim, hidden_dim, output_dim):
        super().__init__()
        self.gat1 = GATConv(input_dim + lri_feat_dim, hidden_dim, heads=2, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_weight, batch):
        h = self.gat1(x, edge_index, edge_weight=edge_weight)
        h = torch.relu(h)
        h = self.gat2(h, edge_index, edge_weight=edge_weight)
        h = global_mean_pool(h, batch)
        return self.decoder(h)
