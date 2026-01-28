# File: src/pipeline/spatial_graph_lri.py
import torch
import pandas as pd
import numpy as np
import scanpy as sc
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def build_spatial_graph(adata, radius=100):
    coords = adata.obsm["spatial"]
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    edge_list = []
    for i, neighbors in enumerate(nbrs.radius_neighbors(coords, return_distance=False)):
        for j in neighbors:
            if i != j:
                edge_list.append((i, j))
    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    return edge_index

def infer_lri_pairs(adata, lri_db: pd.DataFrame, min_expr=0.1):
    lig_expr = adata[:, lri_db["ligand"].unique()].X
    rec_expr = adata[:, lri_db["receptor"].unique()].X
    expressed_ligs = np.array(lig_expr.mean(axis=0)).flatten() > min_expr
    expressed_recs = np.array(rec_expr.mean(axis=0)).flatten() > min_expr

    valid_pairs = lri_db[
        lri_db["ligand"].isin(lri_db["ligand"].unique()[expressed_ligs]) &
        lri_db["receptor"].isin(lri_db["receptor"].unique()[expressed_recs])
    ]
    return valid_pairs

def build_lri_graph(adata, edge_index, lri_df):
    G = nx.DiGraph()
    coords = adata.obsm["spatial"]
    for idx in range(adata.n_obs):
        G.add_node(idx, pos=coords[idx], cell_type=adata.obs.iloc[idx]["cell_type"])

    for _, row in lri_df.iterrows():
        ligand, receptor = row["ligand"], row["receptor"]
        lig_cells = adata[:, ligand].X > 0.5
        rec_cells = adata[:, receptor].X > 0.5
        for i, is_lig in enumerate(lig_cells):
            if is_lig:
                for j, is_rec in enumerate(rec_cells):
                    if is_rec and (i, j) in zip(edge_index[0], edge_index[1]):
                        w = adata[i, ligand].X.item() * adata[j, receptor].X.item()
                        G.add_edge(i, j, ligand=ligand, receptor=receptor, weight=w)
    return G
