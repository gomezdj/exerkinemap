# exerkinemap.py

# =============================
# Imports
# =============================
import os
import warnings
from collections import Counter
from pprint import pprint

import pandas as pd
import numpy as np
import torch
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import gdown

# Spatial tools
import spatialdata as sd
import spatialdata_io as sdio
import spatialdata_plot as sdplot
import napari_spatialdata as nsd
import squidpy as sq
import liana as li

from scvi.model import CellAssign
import torch

# Models and Graphs
from MaxFuse import construct_meta_cells, fuzzy_smoothing, initial_matching, joint_embedding

# HubMAP Client
from hubmap_api_py_client import Client

# HRA Widgets (Assumes installed externally)
from hra_jupyter_widgets import CdeVisualization, NodeDistVis

# Optional placeholder for "em" (ExerkineMap tools, assumed missing)
# import exerkinemap as em  # Uncomment when available

# =============================
# Configuration
# =============================
HUBMAP_ENDPOINT = "https://cells.api.hubmapconsortium.org/api/"
DATA_FOLDER = "data"
CODEX_FILE_URL = "https://datadryad.org/api/v2/files/2572152/download"
CODEX_FILE_PATH = os.path.join(DATA_FOLDER, "23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv")

# =============================
# Warnings Settings
# =============================
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# =============================
# API Client Functions
# =============================
def initialize_hubmap_client():
    client = Client(HUBMAP_ENDPOINT)
    return client


def fetch_hubmap_data(client):
    all_celltypes = client.select_celltypes()
    celltypes = [c["grouping_name"] for c in all_celltypes.get_list()]

    datasets = client.select_datasets(where='celltype', has=celltypes).get_list()
    uuids = [d['uuid'] for d in datasets]

    dataset_cells = {}
    dataset_organ = {}
    dataset_modality = {}

    for uuid in uuids:
        cells_in_dataset = client.select_cells(where='dataset', has=[uuid])
        all_cells = cells_in_dataset.get_list().results_set.get_list()

        population = Counter()
        for cell in all_cells:
            population[cell['cell_type']] += 1
            dataset_organ[uuid] = cell['organ'].lower()
            dataset_modality[uuid] = cell['modality']

        dataset_cells[uuid] = population

    return uuids, dataset_cells, dataset_organ, dataset_modality

# =============================
# Data Handling
# =============================
def ensure_data_folder():
    os.makedirs(DATA_FOLDER, exist_ok=True)


def download_codex_data():
    if not os.path.exists(CODEX_FILE_PATH):
        os.system(f"curl -L {CODEX_FILE_URL} -o {CODEX_FILE_PATH}")
        print(f"Downloaded CODEX file to {CODEX_FILE_PATH}")
    else:
        print(f"CODEX file already exists at {CODEX_FILE_PATH}")


def load_codex_data():
    df = pd.read_csv(CODEX_FILE_PATH, index_col=0)
    return df

# =============================
# Visualization Functions
# =============================
def make_node_list(df, is_3d=False):
    if not is_3d:
        df.loc[:, ('z')] = 0
    return [{'x': row['x'], 'y': row['y'], 'z': row['z'], 'Cell Type': row['Cell Type']} for _, row in df.iterrows()]


def visualize_nodes_2d(node_list, target_celltype='Endothelial'):
    vis = NodeDistVis(
        nodes=node_list,
        node_target_key="Cell Type",
        node_target_value=target_celltype,
        max_edge_distance=1000,
    )
    display(vis)


def visualize_nodes_3d(node_list):
    vis = CdeVisualization(nodes=node_list)
    display(vis)

# =============================
# Modeling Functions
# =============================
def train_scvi_model(source_adata, condition_key, cell_type_key):
    sca.models.SCVI.setup_anndata(source_adata, batch_key=condition_key)
    vae = sca.models.SCVI(
        source_adata,
        n_layers=2,
        encode_covariates=True,
        deeply_inject_covariates=False,
        use_layer_norm="both",
        use_batch_norm="none",
    )
    vae.train()
    return vae


def infer_rna_velocity(adata):
    scv.pp.filter_and_normalize(adata)
    scv.pp.moments(adata)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, basis='umap')


def run_cellassign(adata, marker_dict, label_key="cellassign_labels"):
    """
    Runs CellAssign on an AnnData object.
    
    Args:
        adata (AnnData): Your single-cell or spatial AnnData object.
        marker_dict (dict): Dictionary of {celltype: [list of marker genes]}.
        label_key (str): Where to save CellAssign output labels in adata.obs.
    """
    # Create marker gene matrix
    from cellassign.utils import create_markers_matrix
    
    all_genes = adata.var_names
    marker_mat = create_markers_matrix(marker_gene_dict=marker_dict, all_genes=all_genes)
    
    # Setup AnnData
    CellAssign.setup_anndata(adata)
    
    # Initialize and train model
    model = CellAssign(adata, marker_gene_matrix=marker_mat)
    model.train(max_epochs=250)
    
    # Predict
    preds = model.predict()
    adata.obs[label_key] = preds["assigned_labels"]
    
    print(f"CellAssign completed. Labels stored in adata.obs['{label_key}']")
    return adata, model


def run_liana(adata, groupby='cellassign_labels', resource='consensus'):
    """
    Run LIANA cell-cell communication inference on an AnnData object.
    
    Args:
        adata (AnnData): The AnnData object with cell type labels.
        groupby (str): Column in adata.obs to use as cell types.
        condition_key (str): Column that defines healthy vs disease groups.
        resource (str): Ligand-receptor database (default 'consensus' combines multiple).
        
    Returns:
        adata (AnnData): AnnData with liana results stored.
    """
    adata.uns['liana_results'] = {}

    li.mt.rank_aggregate(adata,
                         groupby=groupby,
                         resource=resource,
                         expr_prop=0.1,  # Minimum % cells expressing ligand/receptor
                         min_cells=5,
                         n_perms=500)  # Permutations for significance estimation
    print(f\"LIANA analysis completed! Results stored in adata.uns['liana'].\")
    return adata

    # adata.uns['liana'] will contain the dataframe of communication scores
    
    df_liana = adata.uns['liana']
    print(df_liana.head())
    
    # You can plot top interactions
    li.pl.dotplot(df_liana,
                  source_labels=['T cells', 'B cells','NK cells'],
                  target_labels=['Fibroblast', 'Cardiomyoctes','CTLs'],
                  top_n=20,
                  figsize=(10, 6))

    for condition in adata.obs[condition_key].unique():
            print(f\"Running LIANA for condition: {condition}\")
            adata_sub = adata[adata.obs[condition_key] == condition].copy()
            
            li.mt.rank_aggregate(adata_sub,
                                 groupby=groupby,
                                 resource=resource,
                                 expr_prop=0.1,
                                 min_cells=5,
                                 n_perms=500)
            
            adata.uns['liana_results'][condition] = adata_sub.uns['liana']
        
        print(\"LIANA analysis completed for all conditions!\")
        return adata


# =============================
# Utilities
# =============================
def load_spatial_data(file_path):
    return sdio.read(file_path)


def plot_spatial_data(spatial_data):
    sdplot.plot(spatial_data)


def preprocess_adata(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    return adata


def compute_spatial_neighbors(adata, spatial_key='spatial'):
    """
    Computes the spatial neighbor graph for the tissue.
    
    Args:
        adata (AnnData): AnnData object with spatial coordinates.
        spatial_key (str): Key where spatial coords are stored.
    """
    sq.gr.spatial_neighbors(adata, coord_type='generic', spatial_key=spatial_key)
    
    gr.interaction_matrix(adata, cluster_key[,...])
    gr.ripley(adata, cluster_key[,mode,...])
    gr.ligrec(adata, cluster_key[,...])
    gr.spatial_autocorr(adata[,...])
    gr.sepal(adata, max_neighs[, genes, n_iter],...)

    im.process(img[, layer, library_id, method, ...])
    im.segment(img[,layer,library_id, method, ...])
    im.calculate_image_features(adata, img[,...])

    pl.spatial_scatter(adata[,shape,color,...])
    pl.spatial_segment(adata[, color, groups, ...])
    pl.nhood_enrichment(adata, cluster_key[,...])
    pl.centrality_scores(adata, cluster_key[,...])
    pl.interaction_matrix(adata, cluster_key[,...])
    pl.ligrec(adata, [, cluster_key,...])
    pl.ripley(adata, cluster_key, ...])
    pl.var_by_distance(adata, var, anchor_key[,...])

    read.visium(path, *[, counts_file, ...])
    read.vizgen(path, *, count_file, meta_file)
    read.nanostring(path, *, counts_file, meta_file)

    tl.var_by_distance(adata, groups[,...])

    datasets.merfish([path])
    datasets.mibitof([path])

    return adata

def run_spatial_liana(adata, groupby='cell_type', condition_key='condition', resource='consensus'):
    """
    Run spatial LIANA based on neighborhoods.
    """
    adata.uns['spatial_liana_results'] = {}

    for condition in adata.obs[condition_key].unique():
        print(f\"Running Spatial LIANA for condition: {condition}\")
        adata_sub = adata[adata.obs[condition_key] == condition].copy()
        
        # Make sure spatial neighbors are computed
        compute_spatial_neighbors(adata_sub)
        
        li.mt.rank_aggregate(adata_sub,
                             groupby=groupby,
                             resource=resource,
                             expr_prop=0.1,
                             min_cells=5,
                             n_perms=500,
                             use_neighbors=True)  # <-- KEY POINT
        
        adata.uns['spatial_liana_results'][condition] = adata_sub.uns['liana']
    
    print(\"Spatial LIANA completed for all conditions!\")
    return adata

def plot_spatial_communication(adata, interaction_of_interest, condition, layer_key='X_spatial'):
    """
    Plots spatial distribution of a specific interaction (ligand-receptor pair).
    
    Args:
        adata (AnnData): AnnData object.
        interaction_of_interest (str): Ligand-Receptor pair name (e.g., 'TGFB1_TGFBR2').
        condition (str): Healthy or disease condition.
        layer_key (str): Data layer for spatial expression (default 'X_spatial').
    """
    results = adata.uns['spatial_liana_results'][condition]
    df = results
    
    # Filter interaction
    selected = df[df['interaction'] == interaction_of_interest]
    
    if selected.empty:
        print(f\"No interaction {interaction_of_interest} found.\")
        return
    
    # Get ligand and receptor genes
    ligand, receptor = selected.iloc[0]['ligand'], selected.iloc[0]['receptor']
    
    # Plot ligand and receptor spatial expression
    sc.pl.spatial(adata, color=[ligand, receptor], layer=layer_key, cmap='viridis')


# =============================
# Main Execution
# =============================
def main():
    ensure_data_folder()
    download_codex_data()

    client = initialize_hubmap_client()
    uuids, dataset_cells, dataset_organ, dataset_modality = fetch_hubmap_data(client)

    df = load_codex_data()
    df_filtered = df[(df['donor'] == "B012") & (df['unique_region'] == "B012_Proximal jejunum")]

    node_list = make_node_list(df_filtered[['x', 'y', 'Cell Type']], is_3d=False)
    pprint(node_list[:5])
    visualize_nodes_2d(node_list)

    marker_dict = {
    ligands: ["WNT","RSPO3","BMP", "GREM1", "WNT5B"],
    immune: ["CD7"],
    "B cells": ["CD20"],
    "Endothelial": ["VWF", "PECAM1"],
    "T cells": ["CD3D", "CD3E", "CD4","CD8"],
    "DC cells": ["CD209", "CD11c"],
    "NK cells": ["CD57"],
    "Smooth muscle": ["MYH11","ACTA2","DES"],
    "Mono/macrophages": ["CD14", "CD16", "CD68", "CD203"],
    "Enteroendocrine": ["CD66", "CD57"],
    "Goblet":[],[],
    "Immature goblet"
    "Enterochromaffin": ["RXFP4", "TPH1"],
    "Enteroendocrine Un"
    "D cells": ["SST"],
    "I cells": ["CCK"],
    "K cells": ["GIP"],
    "L cells": ["INSL5", "GCG", "PYY],
    "Mo cells": ["MLN"],
    "NEUROG3": []
    "S cells": ["SCT"],
    "Sec. Spec. MUC6+": []
    "Stroma": [],
    "Plasma": [],
    "Panth cells":[],[],
    "Tuft": []
    "Stem": []
    "Cycling TA1": [],
    "Cycling TA2": [],
    "TA2": [],
    "TA1": [],
    "Immature ent.": [],
    "Enterocytes":["BEST4"],
    "Enterocytes 2": [],
    "Cajal": ["KIT", "ANO"],
    "glial cells": ["SOX10","CDH19","PLP1"],
    "neurons": ["SYP","SYT1", "RBFOX1"],
    "pericyctes": ["NOTCH3","MCAM1","RGS5"],
    "adipocytes": ["PLIN1","LPL"],
    "Fibroblast": ["COL1A1", "ADAMDEC1", "KCNN3"]
    ],
    # Add more cell types and markers relevant to your system
    }

    adata = sc.read_h5ad("your_data.h5ad")
    adata, cellassign_model = run_cellassign(adata, marker_dict)
    sc.pl.umap(adata, color="cellassign_labels")

    # For 3D version
    df_filtered['z'] = 0
    node_list_3d = make_node_list(df_filtered[['x', 'y', 'z', 'Cell Type']], is_3d=True)
    visualize_nodes_3d(node_list_3d)

    # RNA velocity example
    # adata = sc.read("integrated_data.h5ad")
    # infer_rna_velocity(adata)

    adata = sc.read_h5ad(\"your_dataset.h5ad\")
    adata = run_liana(adata, groupby='cell_type', condition_key='condition')
    adata = run_spatial_liana(adata, groupby='cell_type', condition_key='condition')
    plot_spatial_communication(adata, interaction_of_interest='TNF_TNFR', condition='disease')



if __name__ == "__main__":
    main()
