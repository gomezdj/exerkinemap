import torch
import torch.nn as nn

class ExerkineKineticsEngine(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ExerkineKineticsEngine, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, strain_features: torch.Tensor, gravity_factor: torch.Tensor) -> torch.Tensor:
        """
        Predicts dynamic exerkine secretion rates.
        Inputs:
            strain_features: Tensor of wearable sensor metrics [Batch, Features]
            gravity_factor: Gravitational scaling vector [Batch, 1] (e.g., 0.0 for microgravity, 1.0 for Earth)
        """
        # Modulate input representation by gravitational loading factor
        x = torch.cat([strain_features, gravity_factor], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        secretion_rates = self.fc2(x)
        
        return secretion_rates