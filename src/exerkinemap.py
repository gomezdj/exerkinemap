# Single-cell and Spatial Cell-cell communication frameworks  
# @gomez-dan
import pandas as pd
import scanpy as sc

# Load CSV Data
df = pd.read_csv('path_to_your/exerkinextissue.csv')

# Assuming `df` is loaded from your CSV
adata = sc.AnnData(
    X=df[['Effect']].values,  # The main data matrix can be a placeholder, or any relevant numeric value
    obs=df[['Exerkine', 'Source_Tissue', 'Target_Tissue', 'Biological_System']],
    var=pd.DataFrame(index=df['Exerkine'])
)

adata.write_h5ad('path_to_save/exerkinextissue.h5ad')