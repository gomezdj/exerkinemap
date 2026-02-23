import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNDeconvolution(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNDeconvolution, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

# Define model parameters
input_dim = 128  # Example input dimension
hidden_dim = 64  # Example hidden dimension
output_dim = 10  # Example output dimension

# Initialize model
model = GCNDeconvolution(input_dim, hidden_dim, output_dim)

# Example data
x = torch.randn((100, input_dim))  # Node features
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Edge indices

# Forward pass
output = model(x, edge_index)
print(output)
