adata = sc.read_h5ad(\"your_dataset.h5ad\")
adata = run_liana(adata, groupby='cell_type', condition_key='condition')
adata = run_spatial_liana(adata, groupby='cell_type', condition_key='condition')
plot_spatial_communication(adata, interaction_of_interest='TGFB1_TGFBR2', condition='disease')
