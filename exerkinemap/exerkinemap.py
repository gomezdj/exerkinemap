# exerkinemap.py
import pandas as pd
import scanpy as sc
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

