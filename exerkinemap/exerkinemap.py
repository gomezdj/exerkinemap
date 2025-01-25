# ExerkineMap
# @gomez-dan

import pandas as pd
import scanpy as sc
import os

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

if __name__ == "__main__":
    input_csv_path = 'path_to_your/exerkinextissue.csv'
    output_h5ad_path = 'path_to_save/exerkinextissue.h5ad'
    
    load_and_save_exerkinemap(input_csv_path, output_h5ad_path)
