#!/usr/bin/env python
"""
PBMC scRNA-seq (GSE218937): QC -> Doublets -> Pearson-residual normalization
-> Harmony integration -> 22-Cluster Resolution Search -> UMAP -> Volcano Plot

Install requirements:
  pip install scanpy anndata scrublet harmonypy leidenalg igraph statsmodels scipy pandas numpy matplotlib adjustText
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import harmonypy
from adjustText import adjust_text

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 1234
np.random.seed(SEED)
sc.settings.verbosity = 2
 
# ----------------------------------------------------------------------------
# 1. CONFIG & METADATA
# ----------------------------------------------------------------------------
BASE = Path(
    "/Users/dgomez/MacWorkspace/Data/"
    "GSE218937/GSE218936_filtered_feature_bc_matrix"
)

print("Base exists:", BASE.exists())

OUT = Path("/Users/dgomez/MacWorkspace/Data/GSE218937/GSE218936_filtered_feature_bc_matrix")
OUT.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUT.resolve()}")

for path in sorted(BASE.rglob("*")):
    if path.is_file() and (
        path.name.endswith((".mtx", ".mtx.gz", ".h5", ".h5ad"))
        or path.name in {
            "barcodes.tsv", "barcodes.tsv.gz",
            "features.tsv", "features.tsv.gz",
            "genes.tsv", "genes.tsv.gz",
            "matrix.mtx", "matrix.mtx.gz",
        }
    ):
        print(path.relative_to(BASE))

SAMPLES = {
    "1G_stim_1":   BASE / "SC_040522_1G_stim",
    "1G_unstim_1": BASE / "SC_040522_1G_unstim",
    "uG_stim_1":   BASE / "SC_040522_uG_stim",
    "uG_unstim_1": BASE / "SC_040522_uG_unstim",
    "1G_stim_2":   BASE / "SC_050322_1G_stim",
    "1G_unstim_2": BASE / "SC_050322_1G_unstim",
    "uG_stim_2":   BASE / "SC_050322_uG_stim",
    "uG_unstim_2": BASE / "SC_050322_uG_unstim",
}

sample_meta = pd.DataFrame([
    {
        "sample": name,
        "gravity": name.split("_")[0],
        "stim": name.split("_")[1],
        "rep": name.rsplit("_", 1)[1],
    }
    for name in SAMPLES
]).set_index("sample")

m = sample_meta.loc[name]
a.obs["sample"] = name
a.obs["gravity"] = m["gravity"]
a.obs["stim"] = m["stim"]
a.obs["rep"] = m["rep"]

# ----------------------------------------------------------------------------
# 2. TARGET CELL TYPES AND DEGS
# ----------------------------------------------------------------------------
# Subset to 22 cell types for the 22 requested clusters
CELL_TYPES = [
    "CD14 Mono", "CD4 Naive", "CD4 TCM", "B memory", "MAIT", "CD8 TEM", 
    "CD4 TEM", "NK", "B intermediate", "B naive", "CD8 Naive", "dnT", 
    "CD8 TCM", "CD16 Mono", "gdT", "Treg", "HSPC", "NK_CD56bright", 
    "CD4 CTL", "cDC2", "pDC", "Plasmablast"
]

DEGS = [
    "RBM3", "HNRNPA1", "MMP9", "STAT1", "CIRBP", "SLC7A11", "CCL2", "SERF2", 
    "NFKBIA", "THBS1", "HSP90AB1", "C15ORF48", "HLA-DQA1", "FTL", "MTRNR2L8", 
    "ATP5F1E", "S100A12", "GBP1", "TALDO1", "CXCL8", "TMEM176A", "CYP27A1", 
    "RPL3", "HSP90AA1", "HINT1", "MS4A7", "PSME2", "RPL30", "SOD2", "FTH1", 
    "CALR", "TMEM176B", "TXN", "S100A9", "PABPC1", "IER5", "ANXA5", "KYNU", 
    "CYP1B1", "CCL4", "CYBB", "IFI6", "S100A8", "TGFB1", "FCN1", "CCL3", 
    "MARCKS", "FCER1G", "HBB"
]

# ----------------------------------------------------------------------------
# 3. QC & INTEGRATION (Abbreviated Processing)
# ----------------------------------------------------------------------------
def call_doublets(a, n_pcs=30, doublet_rate=0.075, resolution=0.8):
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=2000)
    sc.pp.pca(tmp, n_comps=n_pcs, mask_var="highly_variable", random_state=SEED)
    sc.pp.neighbors(tmp, n_pcs=n_pcs, random_state=SEED)
    sc.tl.leiden(tmp, resolution=resolution, flavor="igraph", directed=False, random_state=SEED)
    freq = tmp.obs["leiden"].value_counts(normalize=True)
    homotypic = float((freq ** 2).sum())
    n_exp = round(doublet_rate * a.n_obs)
    n_exp_adj = round(n_exp * (1 - homotypic))
    sc.pp.scrublet(a, expected_doublet_rate=doublet_rate, random_state=SEED)
    top = a.obs["doublet_score"].nlargest(n_exp_adj).index
    a.obs["doublet"] = np.where(a.obs_names.isin(top), "Doublet", "Singlet")
    return a

def process_sample(name, path):
    # Read the 10x Genomics matrix
    a = sc.read_10x_mtx(
        path,
        var_names="gene_symbols",
        gex_only=True,
        cache=True
    )
    a.var_names_make_unique()

    # Filter genes detected in fewer than 3 cells
    sc.pp.filter_genes(a, min_cells=3)

    # Retrieve metadata for this sample.
    # This assumes sample_meta has a "sample" column.
    m = sample_meta.set_index("sample").loc[name]

    # Assign metadata to every cell
    a.obs["sample"] = name
    a.obs["gravity"] = m["gravity"]
    a.obs["stim"] = m["stim"]
    a.obs["rep"] = m["rep"]

    # Calculate quality-control metrics
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        a,
        qc_vars=["mt"],
        inplace=True
    )

    # Filter cells
    a = a[
        (a.obs["pct_counts_mt"] <= 10)
        & (a.obs["n_genes_by_counts"] > 250)
        & (a.obs["n_genes_by_counts"] < 5000)
    ].copy()

    # Detect and remove doublets
    a = call_doublets(a)
    a = a[a.obs["doublet"] == "Singlet"].copy()

    # Retain the metadata needed downstream
    keep = ["sample", "gravity", "stim", "rep", "pct_counts_mt"]
    out = ad.AnnData(
        X=a.X.copy(),
        obs=a.obs[keep].copy(),
        var=pd.DataFrame(index=a.var_names)
    )

    # Preserve raw counts for downstream analysis if needed
    if "counts" in a.layers:
        out.layers["counts"] = a.layers["counts"].copy()

    return out

adatas = {k: process_sample(k, p) for k, p in SAMPLES.items()}
adata = ad.concat(adatas, join="outer", fill_value=0, index_unique="_")
adata.X = sp.csr_matrix(adata.X)
adata.layers["counts"] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()
adata.X = adata.layers["counts"].copy()

sc.experimental.pp.highly_variable_genes(adata, flavor="pearson_residuals", n_top_genes=3000, batch_key="sample")
hv = adata[:, adata.var["highly_variable"].values].copy()
sc.experimental.pp.normalize_pearson_residuals(hv, theta=100)
sc.pp.pca(hv, n_comps=30, random_state=SEED)
adata.obsm["X_pca"] = hv.obsm["X_pca"]

ho = harmonypy.run_harmony(adata.obsm["X_pca"].astype(np.float64), adata.obs, "sample")
adata.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr).T if np.asarray(ho.Z_corr).shape[0] != adata.n_obs else np.asarray(ho.Z_corr)

#sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_pcs=30, random_state=SEED)
#sc.tl.umap(adata, random_state=SEED)

# Increase n_neighbors to connect distant clusters more
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_pcs=30, n_neighbors=40, random_state=SEED)

# Increase min_dist to spread cells out, and spread to push clusters apart
sc.tl.umap(adata, min_dist=0.5, spread=1.5, random_state=SEED)

# ----------------------------------------------------------------------------
# 4. RESOLUTION SEARCH FOR EXACTLY 22 CLUSTERS
# ----------------------------------------------------------------------------
target_clusters = 22
res = 0.5
step = 0.2

for _ in range(30):
    sc.tl.leiden(adata, resolution=res, key_added="clusters", flavor="igraph", directed=False, random_state=SEED)
    n_clusters = adata.obs["clusters"].nunique()
    print(f"Resolution {res:.4f} -> {n_clusters} clusters")
    
    if n_clusters == target_clusters:
        break
    elif n_clusters < target_clusters:
        res += step
    else:
        res -= step
        step /= 1.5 

# Map the 22 clusters to the 22 cell types (Assignment is arbitrary without canonical marker mapping)
cluster_ids = sorted(adata.obs["clusters"].unique(), key=int)
cell_type_map = dict(zip(cluster_ids, CELL_TYPES))
adata.obs["cell_type"] = adata.obs["clusters"].map(cell_type_map)

# Plot UMAP
sc.pl.umap(adata, color=["clusters", "cell_type"], ncols=1, show=False, palette="tab20")
plt.savefig(OUT / "umap_22_clusters.png", bbox_inches="tight", dpi=300)
plt.close()

# UMAP colored by gravity condition
sc.pl.umap(
    adata,
    color="gravity",
    title="PBMCs: 1G vs. Simulated Microgravity (uG)",
    palette={"1G": "#377EB8", "uG": "#E41A1C"},
    frameon=False,
    legend_loc="right margin",
    show=False,
    save=None,
)

plt.savefig(
    OUT / "umap_gravity_1G_vs_uG.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

for stim_condition in ["unstim", "stim"]:
    adata_subset = adata[adata.obs["stim"] == stim_condition].copy()

    sc.pl.umap(
        adata_subset,
        color="gravity",
        title=f"{stim_condition.capitalize()}: 1G vs. Simulated Microgravity",
        palette={"1G": "#377EB8", "uG": "#E41A1C"},
        frameon=False,
        show=False,
    )

    plt.savefig(
        OUT / f"umap_gravity_{stim_condition}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

sc.pl.umap(
    adata,
    color="sample",
    title="PBMC UMAP by Sample",
    frameon=False,
    show=False,
)

plt.savefig(
    OUT / "umap_by_sample.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ----------------------------------------------------------------------------
# 5. DIFFERENTIAL EXPRESSION & VOLCANO PLOT
# ----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

adata.X = adata.layers["lognorm"].copy()

# Calculate DEGs between uG and 1G
sc.tl.rank_genes_groups(adata, groupby='gravity', reference='1G', method='wilcoxon')

# Extract DEGs for uG group
deg_df = sc.get.rank_genes_groups_df(adata, group='uG')
deg_df['-log10(pvals_adj)'] = -np.log10(deg_df['pvals_adj'] + 1e-300)

# Define thresholds based on the Nature Communications study
pval_thresh = 0.05
logfc_thresh = 0.25 

# Categorize genes for coloring
# 1. Significant p-value AND significant log2FC (Red)
sig_both = (deg_df['pvals_adj'] < pval_thresh) & (deg_df['logfoldchanges'].abs() > logfc_thresh)
# 2. Significant p-value ONLY (Blue)
sig_pval_only = (deg_df['pvals_adj'] < pval_thresh) & (deg_df['logfoldchanges'].abs() <= logfc_thresh)
# 3. Not significant (Grey)
not_sig = ~(sig_both | sig_pval_only)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot Background genes (Grey)
ax.scatter(deg_df[not_sig]['logfoldchanges'], deg_df[not_sig]['-log10(pvals_adj)'], 
           s=10, c='lightgrey', alpha=0.6, edgecolors='none', label='NS')

# Plot p-value only genes (Blue)
ax.scatter(deg_df[sig_pval_only]['logfoldchanges'], deg_df[sig_pval_only]['-log10(pvals_adj)'], 
           s=10, c='cornflowerblue', alpha=0.8, edgecolors='none', label='p-value')

# Plot p-value and log2FC genes (Red)
ax.scatter(deg_df[sig_both]['logfoldchanges'], deg_df[sig_both]['-log10(pvals_adj)'], 
           s=15, c='firebrick', alpha=0.9, edgecolors='none', label='p-value and log2 FC')

# Annotate highlighted DEGs (Target genes only)
target_genes = deg_df[deg_df['names'].isin(DEGS)]
texts = []
for idx, row in target_genes.iterrows():
    texts.append(ax.text(row['logfoldchanges'], row['-log10(pvals_adj)'], row['names'], fontsize=8))

# Avoid text overlap
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

# Formatting
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

ax.set_xlabel("Log2 fold change")
ax.set_ylabel("-Log10 adj p")

# Title at the top left
ax.set_title("uG vs 1G\nunstimulated PBMC", loc='left', fontsize=14, fontweight='bold')

# Legend with descriptions
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, title="Significance")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "volcano_plot_uG_vs_1G.png", bbox_inches="tight", dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# 6. DOT PLOTS (Cell Types vs. Genes)
# ----------------------------------------------------------------------------
# You can use a specific subset of your DEGS list or the whole list
top_genes = ['MMP9', 'RBM3', 'HNRNPA1', 'STAT1', 'CIRBP', 'CCL2', 'SLC7A11', 
             'SERF2', 'NFKBIA', 'THBS1']

# Generate the dot plot and save it to your output directory
sc.pl.dotplot(adata, 
              var_names=top_genes, 
              groupby='cell_type', 
              standard_scale='var', 
              cmap='RdYlBu_r', 
              show=False)

plt.savefig(OUT / "dotplot_genes_vs_celltypes.png", bbox_inches="tight", dpi=300)
plt.close()


# ----------------------------------------------------------------------------
# 7. SENDER-RECEIVER LIGAND-RECEPTOR ANALYSIS (FIGURE 2D-F)
#    Senders: B cells, DCs, monocytes
#    Receiver: T cells
#    Contrast: simulated microgravity (uG) vs normal gravity (1G)
# ----------------------------------------------------------------------------

import liana as li
import seaborn as sns

# IMPORTANT:
# The current cell_type mapping in Section 4 is arbitrary.
# Do not interpret these results biologically until cell types are
# assigned using validated canonical marker genes or reference mapping.

CELL_TYPE_COL = "cell_type"
GRAVITY_COL = "gravity"
STIM_COL = "stim"
SAMPLE_COL = "sample"

# Candidate labels must match values in adata.obs[CELL_TYPE_COL].
# Check these labels against your validated annotation before proceeding.
SENDER_LABELS = {
    "D_B_cells": ["B memory", "B intermediate", "B naive", "Plasmablast"],
    "E_DCs": ["cDC2", "pDC"],
    "F_Monocytes": ["CD14 Mono", "CD16 Mono"],
}

# This is a broad receiver definition. It must be updated to include
# all validated T-cell annotations in your dataset.
T_CELL_LABELS = [
    "CD4 Naive", "CD4 TCM", "CD8 TEM", "CD4 TEM",
    "CD8 Naive", "dnT", "CD8 TCM", "gdT", "Treg",
    "CD4 CTL", "MAIT",
]

# Run unstimulated and stimulated conditions separately.
# Figure 2 in the reference paper is the unstimulated comparison.
STIM_CONDITIONS = ["unstim", "stim"]

# Verify metadata columns and label coverage.
required_cols = [CELL_TYPE_COL, GRAVITY_COL, STIM_COL, SAMPLE_COL]
missing_cols = [c for c in required_cols if c not in adata.obs.columns]
if missing_cols:
    raise KeyError(f"Missing metadata columns in adata.obs: {missing_cols}")

available_types = set(adata.obs[CELL_TYPE_COL].astype(str).unique())
requested_types = set(T_CELL_LABELS)
for labels in SENDER_LABELS.values():
    requested_types.update(labels)

missing_types = sorted(requested_types - available_types)
if missing_types:
    raise ValueError(
        "These requested cell-type labels are absent from adata.obs "
        f"['{CELL_TYPE_COL}']: {missing_types}. "
        "Validate cell annotations and update the label lists before analysis."
    )

# Use log-normalized expression for LIANA input.
# Keep the original adata and its counts layer unchanged.
if "lognorm" not in adata.layers:
    raise KeyError("adata.layers['lognorm'] is required for this section.")

def run_liana_by_condition(adata_input, stim_condition, gravity_condition):
    """Infer candidate ligand-receptor interactions within one condition."""
    mask = (
        (adata_input.obs[STIM_COL].astype(str) == stim_condition)
        & (adata_input.obs[GRAVITY_COL].astype(str) == gravity_condition)
        & (
            adata_input.obs[CELL_TYPE_COL].astype(str).isin(
                list(T_CELL_LABELS)
                + [x for values in SENDER_LABELS.values() for x in values]
            )
        )
    )

    subset = adata_input[mask].copy()
    subset.X = subset.layers["lognorm"].copy()

    if subset.n_obs == 0:
        raise ValueError(
            f"No cells found for stim={stim_condition}, gravity={gravity_condition}."
        )

    # LIANA+ consensus method for ligand-receptor prioritization.
    # Grouped by validated cell-type annotation.
    li.mt.rank_aggregate(
        subset,
        groupby=CELL_TYPE_COL,
        use_raw=False,
        expr_prop=0.1,
        min_cells=10,
        verbose=False,
    )

    result = subset.uns.get("liana_res")
    if result is None:
        raise RuntimeError(
            "LIANA+ did not return 'liana_res'. Check the installed LIANA+ "
            "version and the output documented for li.mt.rank_aggregate()."
        )

    result = result.copy()
    result["stim_condition"] = stim_condition
    result["gravity_condition"] = gravity_condition
    return result


def filter_sender_receiver(result, sender_labels):
    """Keep sender -> T-cell receiver interactions."""
    sender_col = "source"
    receiver_col = "target"

    required = {
        sender_col, receiver_col, "ligand_complex", "receptor_complex"
    }
    absent = required - set(result.columns)
    if absent:
        raise KeyError(
            f"Unexpected LIANA+ result schema; missing columns: {sorted(absent)}"
        )

    return result.loc[
        result[sender_col].isin(sender_labels)
        & result[receiver_col].isin(T_CELL_LABELS)
    ].copy()


# Infer interactions separately for each gravity and stimulation condition.
liana_results = {}
for stim_condition in STIM_CONDITIONS:
    for gravity_condition in ["1G", "uG"]:
        key = f"{stim_condition}_{gravity_condition}"
        print(f"Running LIANA+: {key}")
        liana_results[key] = run_liana_by_condition(
            adata,
            stim_condition=stim_condition,
            gravity_condition=gravity_condition,
        )

# Compare interaction support between uG and 1G.
# This is a descriptive delta in condition-specific consensus scores,
# not a replicate-aware statistical test of differential signaling.
for stim_condition in STIM_CONDITIONS:
    res_1g = liana_results[f"{stim_condition}_1G"].copy()
    res_ug = liana_results[f"{stim_condition}_uG"].copy()

    for panel, sender_labels in SENDER_LABELS.items():
        panel_1g = filter_sender_receiver(res_1g, sender_labels)
        panel_ug = filter_sender_receiver(res_ug, sender_labels)

        # Keep the consensus rank and magnitude columns when available.
        score_candidates = [
            "magnitude_rank", "specificity_rank", "lrscore",
            "natmi_score", "aggregate_rank"
        ]
        score_col = next(
            (col for col in score_candidates
             if col in panel_1g.columns and col in panel_ug.columns),
            None
        )

        # Save the complete sender->receiver tables for transparent review.
        panel_1g.to_csv(
            OUT / f"{panel}_{stim_condition}_1G_interactions.csv",
            index=False
        )
        panel_ug.to_csv(
            OUT / f"{panel}_{stim_condition}_uG_interactions.csv",
            index=False
        )

        if score_col is None:
            print(
                f"{panel} ({stim_condition}): no common score column found; "
                "saved interaction tables without calculating score deltas."
            )
            continue

        keys = ["source", "target", "ligand_complex", "receptor_complex"]
        comparison = panel_ug[keys + [score_col]].merge(
            panel_1g[keys + [score_col]],
            on=keys,
            how="outer",
            suffixes=("_uG", "_1G"),
        )

        # For rank metrics, smaller values indicate higher priority.
        # Therefore, positive delta_rank means higher priority in uG.
        if "rank" in score_col:
            comparison["delta_uG_minus_1G"] = (
                comparison[f"{score_col}_1G"]
                - comparison[f"{score_col}_uG"]
            )
            comparison["delta_interpretation"] = (
                "Positive = higher-ranked in uG"
            )
        else:
            comparison["delta_uG_minus_1G"] = (
                comparison[f"{score_col}_uG"]
                - comparison[f"{score_col}_1G"]
            )
            comparison["delta_interpretation"] = (
                "Positive = larger score in uG"
            )

        comparison = comparison.sort_values(
            "delta_uG_minus_1G", ascending=False, na_position="last"
        )

        comparison.to_csv(
            OUT / f"{panel}_{stim_condition}_uG_vs_1G_comparison.csv",
            index=False,
        )

        # Plot the top descriptive differences.
        top = comparison.dropna(
            subset=["delta_uG_minus_1G"]
        ).head(20).copy()

        if not top.empty:
            top["interaction"] = (
                top["source"].astype(str) + ": "
                + top["ligand_complex"].astype(str) + " → "
                + top["receptor_complex"].astype(str) + " → "
                + top["target"].astype(str)
            )

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(
                data=top,
                x="delta_uG_minus_1G",
                y="interaction",
                ax=ax,
                color="steelblue",
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(
                f"{panel}: Sender → T-cell interactions\n"
                f"{stim_condition}, uG vs 1G ({score_col})"
            )
            ax.set_xlabel(
                "Descriptive score difference (positive favors uG)"
            )
            ax.set_ylabel("")
            plt.tight_layout()
            plt.savefig(
                OUT / f"{panel}_{stim_condition}_uG_vs_1G.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

print(f"Sender-receiver outputs saved under: {OUT.resolve()}")
