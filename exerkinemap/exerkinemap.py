# exerkinemap.py
import os
os.chidir('../')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import scanpy as sc
import torch
import scarches as sca 
from sccarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np
import gdown 

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

import scvelo as scv
import spatialdata as sd
import spatialdata_io as sdio
import spatialdata_plot as sdplot
import napari_spatialdata as nsd
import squidpy as sq
import os
from MaxFuse import construct_meta_cells, fuzzy_smoothing, initial_matching, joint_embedding
from SPACEc import delaunay_edges, compute_distances
from STELLAR import STELLAR, construct_graph, build_adjacency_matrix

url = 'https://drive.google.com/file/d/1-S16mXzy19ITG9mbAJMna5jCb-tYo_1F/view?usp=drive_link'
adata = sc.read_csv('celltype_annotations.csv', first_column_names = True)
gdown.download(url, output, quiet=False)

# Save as an H5AD file if needed
# adata.write("celltype_annotations.h5ad")
# adata_all = sc.read('.h5ad')

adata = adata_all.raw.to_adata()
adata = remove_sparsity(adata)
source_adata = adata[~adata.obs[condition_key].isin(target_conditions)].copy()
target_adata = adata[adata.obs[condition_key].isin(target_conditions)].copy()

source_adata
target_adata

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


reference_latent = sc.AnnData(vae.get_latent_representation())
reference_latent.obs["cell_type"] = source_adata.obs[cell_type_key].tolist()
reference_latent.obs["batch"] = source_adata.obs[condition_key].tolist()


sc.pp.neighbors(reference_latent, n_neighbors=8)
sc.tl.leiden(reference_latent)
sc.tl.umap(reference_latent)
sc.pl.umap(reference_latent,
           color=['batch', 'cell_type'],
           frameon=False,
           wspace=0.6,
           )

ref_path = 'ref_model/'
vae.save(ref_path, overwrite=True)

model = sca.models.SCVI.load_query_data(
    target_adata,
    ref_path,
    freeze_dropout = True,
)

model.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0))

query_latent = sc.AnnData(model.get_latent_representation())
query_latent.obs['cell_type'] = target_adata.obs[cell_type_key].tolist()
query_latent.obs['batch'] = target_adata.obs[condition_key].tolist()

sc.pp.neighbors(query_latent)
sc.tl.leiden(query_latent)
sc.tl.umap(query_latent)
plt.figure()
sc.pl.umap(
    query_latent,
    color=["batch", "cell_type"],
    frameon=False,
    wspace=0.6,
)

surgery_path = 'surgery_model'
model.save(surgery_path, overwrite=True)

adata_full = source_adata.concatenate(target_adata)
full_latent = sc.AnnData(model.get_latent_representation(adata=adata_full))
full_latent.obs['cell_type'] = adata_full.obs[cell_type_key].tolist()
full_latent.obs['batch'] = adata_full.obs[condition_key].tolist()

sc.pp.neighbors(full_latent)
sc.tl.leiden(full_latent)
sc.tl.umap(full_latent)
plt.figure()
sc.pl.umap(
    full_latent,
    color=["batch", "cell_type"],
    frameon=False,
    wspace=0.6,
)

def load_and_save_exerkinemap(input_csv_path, output_h5ad_path):
    try:
        # Load CSV Data
        df = pd.read_csv(input_csv_path)

        # Create AnnData object from DataFrame
        adata = sc.AnnData(
            X=df[['Effect']].values,  # The main data matrix can be a placeholder, or any relevant numeric value
            obs=df[['Exerkine', 'Source_Tissue', 'Target_Tissue', 'Biological_System']],
            var=pd.DataFrame(index=df['Exerkine'])
        )
        
        # Save AnnData object to .h5ad format
        adata.write_h5ad(output_h5ad_path)
        print(f"Data successfully saved to {output_h5ad_path}")
    
    except FileNotFoundError:
        print(f"File not found: {input_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Load and preprocess your data
transcriptomics_data = em.load_data("transcriptomics.csv")
proteomics_data = em.load_data("proteomics.csv")
spatial_data = em.load_spatial_data("spatial_data.csv")
singlecell_data = em.load_singlecell_data("singlecell_data.csv")

# Integrate multiomics data
integrated_data = em.integrate_data([transcriptomics_data, proteomics_data])

# Run analysis
exerkine_profiles = em.identify_exerkines(integrated_data)
trajectories = em.infer_trajectories(integrated_data)
spatial_map = em.map_spatial_relationships(spatial_data, integrated_data)

# Visualize the results
em.plot_heatmap(exerkine_profiles)
em.plot_umap(trajectories)
em.plot_spatial_map(spatial_map)

# Additional analysis with scverse
adata = sc.read("integrated_data.h5ad")

# Preprocess for RNA velocity
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata)

# Compute RNA velocity
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)

# Visualize RNA velocity
scv.pl.velocity_embedding_stream(adata, basis='umap')

def load_spatial_data(file_path):
    try:
        spatial_data = sdio.read(file_path)
        return spatial_data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_spatial_data(spatial_data):
    try:
        sdplot.plot(spatial_data)
    except Exception as e:
        print(f"An error occurred while plotting: {e}")

def preprocess_data(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    return adata

def find_spatially_variable_genes(adata):
    sc.tl.rank_genes_groups(adata, 'spatial', method='wilcoxon')
    return adata

def plot_spatial_genes(adata):
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

def run_exerkinemap():
    input_csv_path = 'path_to_your/exerkinextissue.csv'
    output_h5ad_path = 'path_to_save/exerkinextissue.h5ad'
    
    load_and_save_exerkinemap(input_csv_path, output_h5ad_path)

    # Load and plot spatial data
    spatial_file_path = 'path_to_your/spatial_data_file'
    spatial_data = load_spatial_data(spatial_file_path)
    plot_spatial_data(spatial_data)

    # Interactive exploration using napari
    viewer = nsd.view(spatial_data)

if __name__ == "__main__":
    run_exerkinemap()

