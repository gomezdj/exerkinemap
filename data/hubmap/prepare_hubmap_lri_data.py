# File: src/data/prepare_hubmap_lri_data.py

import torch
import pandas as pd
import scanpy as sc
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def build_edge_index(coords, k=6):
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    edge_list = []
    for i, nbrs in enumerate(nn.kneighbors(coords, return_distance=False)):
        for j in nbrs:
            if i != j:
                edge_list.append([i, j])
    return torch.tensor(edge_list, dtype=torch.long).T

def create_pyg_data_from_hubmap(adata: sc.AnnData, lri_node_feats: pd.DataFrame, lri_edge_weights: pd.DataFrame, k=6):
    coords = adata.obsm["spatial"]
    gene_expr = adata.X
    lri_feats = lri_node_feats.loc[adata.obs_names].values

    x = torch.tensor(
        pd.DataFrame(gene_expr.toarray(), index=adata.obs_names)
        .join(pd.DataFrame(lri_feats, index=adata.obs_names))
        .values, dtype=torch.float
    )

    edge_index = build_edge_index(coords, k=k)

    edge_weight = []
    for i, j in zip(edge_index[0], edge_index[1]):
        cid_i = adata.obs_names[i]
        cid_j = adata.obs_names[j]
        w = lri_edge_weights.loc[cid_i, cid_j] if cid_i in lri_edge_weights.index and cid_j in lri_edge_weights.columns else 0.0
        edge_weight.append(w)

    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    y = torch.tensor(adata.obs['cell_type'].astype('category').cat.codes.values, dtype=torch.long)
    batch = torch.zeros(len(y), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, batch=batch)
