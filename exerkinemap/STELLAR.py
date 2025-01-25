import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class STELLAR(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(STELLAR, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gcn = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.fc1(x))
        x = self.gcn_forward(x, adj)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def gcn_forward(self, x, adj):
        out = self.fc2(torch.spmm(adj, x))
        return out

    def loss(self, output, labels, adj, y_true):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        novel_loss = self.novel_loss(output, y_true, adj)
        reg = self.regularization(output)
        return loss + novel_loss + reg

    def novel_loss(self, output, y_true, adj):
        # Assuming pseudo-labels for novel classes
        pred = torch.argmax(output, dim=1)
        mask = (y_true == -1)  # Indices for novel cells
        labeled = pred[~mask]
        unlabeled = pred[mask]
        return F.kl_div(labeled, unlabeled)

    def regularization(self, output):
        return torch.mean(torch.sum(output * torch.log(output + 1e-9), dim=1))

from scipy.spatial import Delaunay

def construct_graph(X, tau):
    tri = Delaunay(X)
    edges = []
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                d = np.linalg.norm(X[simplex[i]] - X[simplex[j]])
                if d < tau:
                    edges.append((simplex[i], simplex[j]))
    return edges

def build_adjacency_matrix(edges, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        adj[edge[0], edge[1]] = 1
        adj[edge[1], edge[0]] = 1
    return torch.FloatTensor(adj)
