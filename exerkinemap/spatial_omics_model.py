# File: src/models/spatial_omics_model.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

class SpatialOmicsModel(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        spatial_dim: int,
        lri_feat_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_gene_network: bool = False
    ):
        super().__init__()
        self.use_protein_network = use_protein_network

        # Optional protein-level network encoder
        if self.use_protein_network:
            self.protein_encoder = GCNConv(protein_dim, protein_dim)  # input = protein graph

        # Cell-level encoder: combines protein, spatial, and LRI
        self.encoder1 = GATConv(protein_dim + spatial_dim + lri_feat_dim, hidden_dim, heads=2)
        self.encoder2 = GATConv(hidden_dim * 2, hidden_dim, heads=1)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, protein_expr, spatial_pos, lri_feats, edge_index, batch, protein_edge_index=None):
        # Optional: refine protein features using protein network
        if self.use_protein_network and protein_edge_index is not None:
            protein_expr = self.protein_encoder(protein_expr, protein_edge_index)

        # Concatenate all modalities
        x = torch.cat([protein_expr, spatial_pos, lri_feats], dim=1)

        h = self.encoder1(x, edge_index)
        h = self.encoder2(h, edge_index)

        h_pool = global_mean_pool(h, batch)
        return self.decoder(h_pool)
    
import pandas as pd

# ...existing imports...

class Fusor:
    # ...existing code...

    def construct_spatial_graphs_for_gnn(
        self,
        coords1: np.ndarray,
        protein1: np.ndarray,
        lri_feats1: np.ndarray,
        coords2: np.ndarray,
        protein2: np.ndarray,
        lri_feats2: np.ndarray,
        cell_metadata1: pd.DataFrame,
        cell_metadata2: pd.DataFrame,
        lr_db: dict,
        n_neighbors: int = 15,
        radius: float = None,
        use_features: bool = True,
        verbose: bool = True
    ):
        """
        Construct spatial graphs for both datasets, annotate with LRI, and prepare for GNN input.

        Parameters
        ----------
        coords1, coords2: np.ndarray
            Cell centroid coordinates for arr1 and arr2.
        protein1, protein2: np.ndarray
            Protein expression features for arr1 and arr2.
        lri_feats1, lri_feats2: np.ndarray
            LRI features for arr1 and arr2.
        cell_metadata1, cell_metadata2: pd.DataFrame
            Metadata with ligand/receptor columns.
        lr_db: dict
            Ligand-receptor database.
        n_neighbors: int
            Number of neighbors for kNN graph.
        radius: float, optional
            Restrict edges to within this spatial radius.
        use_features: bool
            If True, use features+coords; else, only coords.
        verbose: bool
            Print progress.

        Returns
        -------
        graph_data1, graph_data2: dict
            Each dict contains: node_features, edge_index, edge_attr, edge_lri_annotations
        """
        # 1. Build spatial graphs
        rows1, cols1, vals1 = graph.construct_spatial_graph(
            coords=coords1, features=protein1, n_neighbors=n_neighbors, radius=radius,
            use_features=use_features, verbose=verbose
        )
        rows2, cols2, vals2 = graph.construct_spatial_graph(
            coords=coords2, features=protein2, n_neighbors=n_neighbors, radius=radius,
            use_features=use_features, verbose=verbose
        )

        # 2. Annotate edges with LRI pairs
        edge_lri1 = graph.annotate_ligand_receptor_edges(
            rows=rows1, cols=cols1, cell_metadata=cell_metadata1,
            ligand_col='ligand', receptor_col='receptor', lr_db=lr_db
        )
        edge_lri2 = graph.annotate_ligand_receptor_edges(
            rows=rows2, cols=cols2, cell_metadata=cell_metadata2,
            ligand_col='ligand', receptor_col='receptor', lr_db=lr_db
        )

        # 3. Prepare node features for GNN (concat protein, spatial, LRI)
        node_features1 = np.concatenate([protein1, coords1, lri_feats1], axis=1)
        node_features2 = np.concatenate([protein2, coords2, lri_feats2], axis=1)

        # 4. Prepare edge_index and edge_attr for PyTorch Geometric
        edge_index1 = np.stack([rows1, cols1], axis=0)
        edge_index2 = np.stack([rows2, cols2], axis=0)
        edge_attr1 = vals1
        edge_attr2 = vals2

        graph_data1 = {
            'node_features': node_features1,
            'edge_index': edge_index1,
            'edge_attr': edge_attr1,
            'edge_lri_annotations': edge_lri1
        }
        graph_data2 = {
            'node_features': node_features2,
            'edge_index': edge_index2,
            'edge_attr': edge_attr2,
            'edge_lri_annotations': edge_lri2
        }
        if verbose:
            print(f"Graph 1: {edge_index1.shape[1]} edges, {node_features1.shape[0]} nodes")
            print(f"Graph 2: {edge_index2.shape[1]} edges, {node_features2.shape[0]} nodes")
        return graph_data1, graph_data2

    # ...existing Fusor methods...
