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
from pprint import pprint

from SPACEc import delaunay_edges, compute_distances
from STELLAR import STELLAR, construct_graph, build_adjacency_matrix
from hubmap_api_py_client import Client
from collections import Counter

endpoint_url = "https://cells.api.hubmapconsortium.org/api/"
client = Client(endpoint_url) 

all_celltypes = client.select_celltypes()
assert len(all_celltypes) > 0

celltypes = [c["grouping_name"] for c in all_celltypes.get_list()]
print('cell types:', len(celltypes))

# Find all daasets that have been annotated with cell types
datasets = client.select_datasets(where='celltype', has=celltypes).get_list()
assert len(datasets) > 0

uuids = [ d['uuid'] for d in datasets ]
print('annotated datasets with cell types:', len(datasets))

# Get cells for each annotated dataset
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

# Show raw data results for one dataset
print(uuids[0], 'top cell types:', dataset_cells[uuids[0]].most_common(5))
print(uuids[0], 'organ:', dataset_organ[uuids[0]])
print(uuids[0], 'modality:', dataset_modality[uuids[0]])

sum(( sum(pop.values()) for pop in dataset_cells.values() ))

set(dataset_modality.values())

# Make sure the data folder is present
folder_path = "data"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")

# Define the path to the file. 
file_path = f'{folder_path}/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv'

# Check if the file exists
if not os.path.exists(file_path):
    # If the file doesn't exist, run the curl command
    !curl -L https://datadryad.org/api/v2/files/2572152/download -o {file_path}
    print(f"File downloaded and saved at {file_path}")
else:
    print(f"File already exists at {file_path}")

#Install and import external packages
%pip install matplotlib pandas ipywidgets hra_jupyter_widgets

import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets

# Import hra-jupyter-widgets. For documentation, please see https://github.com/x-atlas-consortia/hra-jupyter-widgets/blob/main/usage.ipynb
from hra_jupyter_widgets import (
    BodyUi,
    CdeVisualization, # in this example, we will use this one...
    Eui,
    EuiOrganInformation,
    FtuExplorer,
    FtuExplorerSmall,
    MedicalIllustration,
    ModelViewer,
    NodeDistVis, # ...and this one, but all of them are usable for different purposes!
    Rui,
)

# Make sure the data folder is present
folder_path = "data"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")

# Define the path to the file. 
file_path = f'{folder_path}/23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv'

# Check if the file exists
if not os.path.exists(file_path):
    # If the file doesn't exist, run the curl command
    !curl -L https://datadryad.org/api/v2/files/2572152/download -o {file_path}
    print(f"File downloaded and saved at {file_path}")
else:
    print(f"File already exists at {file_path}")

# Read the CSV file and convert it to a df
df = pd.read_csv(file_path, index_col=0)
df

# Only keep cells from one dataset by selecting 1 donor and 1 region
df_filtered = df[(df['donor'] == "B012") & (
    df['unique_region'] == "B012_Proximal jejunum")]

# Make new df with only x, y, and Cell Type columns (needed for node-dist-vis)
df_cells = df_filtered[['x', 'y', 'Cell Type']]
df_cells

# Next, let's define a function that turns a DataFrame into a node list that can then be passed into the CdeVisualization or NodeDistVis widget
def make_node_list(df:pd.DataFrame, is_3d:bool = False):
  """Turn a DataFrame into a list of dicts for passing them into a HRA widget

  Args:
      df (pd.DataFrame): A DataFrame with cells
  """
  
  # If the df does not have a z-axis column, let's add one and set all cells to 0
  if not is_3d:
    df.loc[:, ('z')] = 0
  
  node_list = [{'x': row['x'], 'y': row['y'], 'z': row['z'], 'Cell Type': row['Cell Type']}
                 for index, row in df.iterrows()]

  return node_list
  
# Prepare df_cells for visualization with NodeDistVis widget
node_list = make_node_list(df_cells, False)

# Let's inspect the first 5 rows
pprint(node_list[:5])

# Finally, let's instantiate the NodeDistVis class with some parameters. We pass in the node_list, indicate Endothelial cells as targets for the edges. 
# As we are not supplying an edge list, we need to provide a max_edge_distance, which is set to 1000 (generiously)
node_dist_vis = NodeDistVis(
    nodes = node_list,
    node_target_key="Cell Type",
    node_target_value="Endothelial",
    max_edge_distance = 1000
)

# Display our new widget
display(node_dist_vis)

df_filtered

# Create a new data frame with values from the NodeDistVis example
df = df_filtered

# indicate the number of layers you would like to show. In our case, let's show 3.
n_layers = 3

# Create a list to hold all the data frames
df_list = [df]

for i in range(0, n_layers-1):

  # Create a copy of this new DataFrame
  df_copy = df.copy()

  # Modify a column in the copied rows (e.g., change values in column 'B')
  df_copy['unique_region'] = f'copy_{i}'  

  # Add df_copy to list of df
  df_list.append(df_copy)
  
# Concatenate the original DataFrame with the modified copies
df_combined = pd.concat(df_list, ignore_index=True)

df_filtered_3d = df_combined

# Set a z-offset
offset = 1000

# Set z axis (or any other axis) by region
df_filtered_3d['z'] = df_filtered_3d['unique_region'].apply(lambda v: 0 if v == 'B012_Proximal jejunum'
                                                            else offset if v == 'copy_0'
                                                            else offset * 2)

# Make new df with only x, y, z, and Cell Type columns
df_cells_3d = df_filtered_3d[['x', 'y', 'z','Cell Type']]

# Prepare df_cells_3d for visualization with CdeVisualization widget
node_list = make_node_list(df_cells_3d, True)

# Let's inspect the first 5 rows
pprint(node_list[:5])

# Finally, let's instantiate the CDEVisualization class with our node_list as parameter.
cde = CdeVisualization(
    nodes=node_list
)

# Display our new widget
display(cde)

# Load Azimuth Reference Dataset
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

