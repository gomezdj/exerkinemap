# File: src/training/train_lri_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score
from src.models.lri_spatial_gnn import LRISpatialGNN

class SpatialGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=2, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, batch):
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = self.gat2(h, edge_index)
        h = global_mean_pool(h, batch)
        return self.decoder(h)

# Training loop
def train_spatial_gnn(model, dataset, epochs=30, lr=1e-3, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    return model

# Evaluation
def evaluate_spatial_gnn(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = DataLoader(dataset, batch_size=32)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(batch.y.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

def train_model(dataset, input_dim, output_dim, epochs=30, lr=1e-3):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LRISpatialGNN(input_dim, hidden_dim=64, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    return model

def evaluate(model, dataset):
    loader = DataLoader(dataset, batch_size=32)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=1)
            y_pred.extend(preds.tolist())
            y_true.extend(batch.y.tolist())
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
