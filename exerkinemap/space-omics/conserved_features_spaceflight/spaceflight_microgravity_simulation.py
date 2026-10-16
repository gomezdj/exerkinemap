#!/usr/bin/env python
"""
PBMC scRNA-seq (GSE218937): QC -> doublet detection -> Pearson-residual HVGs
-> Harmony integration -> Leiden resolution search -> UMAP -> exploratory DE
-> LIANA+ sender-receiver analysis.

Install requirements:
  pip install scanpy anndata scrublet harmonypy leidenalg igraph statsmodels scipy pandas numpy matplotlib adjustText seaborn liana

Important:
- Set BASE to the directory containing the SC_* sample directories.
- Cell-type labels below are NOT assigned from cluster IDs. Biological
  interpretation requires canonical-marker or reference-based annotation.
- Differential-expression and LIANA comparisons are exploratory. The
  condition comparisons here do not model biological replicates.
"""

import warnings
from pathlib import Path

import anndata as ad
import harmonypy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from adjustText import adjust_text
import liana as li

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 1234
np.random.seed(SEED)
sc.settings.verbosity = 2

# ----------------------------------------------------------------------------
# 1. CONFIG & METADATA
# ----------------------------------------------------------------------------
# BASE must contain the SC_* sample directories (not one sample directory).
BASE = Path(
    "/Users/dgomez/MacWorkspace/Data/"
    "GSE218937/GSE218936_filtered_feature_bc_matrix"
)
OUT = Path("/Users/dgomez/MacWorkspace/Data/GSE218937/pbmc_out")
OUT.mkdir(parents=True, exist_ok=True)

print("Base exists:", BASE.exists())
print(f"Input directory: {BASE.resolve()}")
print(f"Output directory: {OUT.resolve()}")

if not BASE.is_dir():
    raise FileNotFoundError(f"Input directory does not exist: {BASE}")

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

sample_meta = pd.DataFrame(
    [
        {
            "sample": name,
            "gravity": name.split("_")[0],
            "stim": name.split("_")[1],
            "rep": name.rsplit("_", 1)[1],
        }
        for name in SAMPLES
    ]
).set_index("sample")

# Target labels are retained as a reference list only. Do not map cluster IDs
# to these labels arbitrarily; annotate clusters using markers/reference data.
CELL_TYPES = [
    "CD14 Mono", "CD4 Naive", "CD4 TCM", "B memory", "MAIT", "CD8 TEM",
    "CD4 TEM", "NK", "B intermediate", "B naive", "CD8 Naive", "dnT",
    "CD8 TCM", "CD16 Mono", "gdT", "Treg", "HSPC", "NK_CD56bright",
    "CD4 CTL", "cDC2", "pDC", "Plasmablast",
]

DEGS = [
    "RBM3", "HNRNPA1", "MMP9", "STAT1", "CIRBP", "SLC7A11", "CCL2", "SERF2",
    "NFKBIA", "THBS1", "HSP90AB1", "C15ORF48", "HLA-DQA1", "FTL", "MTRNR2L8",
    "ATP5F1E", "S100A12", "GBP1", "TALDO1", "CXCL8", "TMEM176A", "CYP27A1",
    "RPL3", "HSP90AA1", "HINT1", "MS4A7", "PSME2", "RPL30", "SOD2", "FTH1",
    "CALR", "TMEM176B", "TXN", "S100A9", "PABPC1", "IER5", "ANXA5", "KYNU",
    "CYP1B1", "CCL4", "CYBB", "IFI6", "S100A8", "TGFB1", "FCN1", "CCL3",
    "MARCKS", "FCER1G", "HBB",
]


# ----------------------------------------------------------------------------
# 2. QC & INTEGRATION
# ----------------------------------------------------------------------------
def call_doublets(a, n_pcs=30, doublet_rate=0.075, resolution=0.8):
    """Run Scrublet; use a temporary normalized copy for homotypic adjustment."""
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=min(2000, tmp.n_vars))
    n_comps = min(n_pcs, max(2, tmp.n_obs - 1), max(2, int(tmp.var["highly_variable"].sum()) - 1))
    sc.pp.pca(tmp, n_comps=n_comps, mask_var="highly_variable", random_state=SEED)
    sc.pp.neighbors(tmp, n_pcs=n_comps, random_state=SEED)
    sc.tl.leiden(
        tmp, resolution=resolution, flavor="igraph",
        directed=False, random_state=SEED,
    )
    freq = tmp.obs["leiden"].value_counts(normalize=True)
    homotypic = float((freq ** 2).sum())
    n_exp = round(doublet_rate * a.n_obs)
    n_exp_adj = round(n_exp * (1 - homotypic))

    # Scrublet operates on the input's count matrix. At this point a.X is raw.
    sc.pp.scrublet(a, expected_doublet_rate=doublet_rate, random_state=SEED)
    scores = a.obs["doublet_score"].fillna(-np.inf)
    top = scores.nlargest(min(n_exp_adj, int(np.isfinite(scores).sum()))).index
    a.obs["doublet"] = np.where(a.obs_names.isin(top), "Doublet", "Singlet")
    return a


def process_sample(name, path):
    """Load and QC one 10x sample; preserve raw counts in layers['counts']."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"Sample directory not found for {name}: {path}\n"
            "Check BASE and the actual directory structure. The path supplied "
            "to read_10x_mtx must contain matrix.mtx(.gz), barcodes.tsv(.gz), "
            "and features.tsv(.gz) (or genes.tsv(.gz))."
        )

    a = sc.read_10x_mtx(
        path,
        var_names="gene_symbols",
        gex_only=True,
        cache=True,
    )
    a.var_names_make_unique()

    # Preserve raw counts explicitly before any normalization.
    a.layers["counts"] = a.X.copy()
    sc.pp.filter_genes(a, min_cells=3)

    m = sample_meta.loc[name]
    a.obs["sample"] = name
    a.obs["gravity"] = m["gravity"]
    a.obs["stim"] = m["stim"]
    a.obs["rep"] = m["rep"]

    a.var["mt"] = a.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True)

    a = a[
        (a.obs["pct_counts_mt"] <= 10)
        & (a.obs["n_genes_by_counts"] > 250)
        & (a.obs["n_genes_by_counts"] < 5000)
    ].copy()

    if a.n_obs == 0:
        raise ValueError(f"No cells remain after QC for sample {name}.")

    a = call_doublets(a)
    a = a[a.obs["doublet"] == "Singlet"].copy()

    if a.n_obs == 0:
        raise ValueError(f"No singlets remain after doublet filtering for {name}.")

    # Keep metadata, doublet score/class, and QC metrics for auditability.
    keep = [
        "sample", "gravity", "stim", "rep", "pct_counts_mt",
        "n_genes_by_counts", "total_counts", "doublet", "doublet_score",
    ]
    keep = [col for col in keep if col in a.obs.columns]
    out = ad.AnnData(
        X=a.layers["counts"].copy(),
        obs=a.obs[keep].copy(),
        var=a.var.copy(),
    )
    out.layers["counts"] = out.X.copy()
    return out


adatas = {name: process_sample(name, path) for name, path in SAMPLES.items()}
adata = ad.concat(
    adatas, join="outer", fill_value=0, index_unique="_",
)
adata.X = sp.csr_matrix(adata.layers["counts"].copy())
adata.layers["counts"] = adata.X.copy()

if adata.n_obs == 0 or adata.n_vars == 0:
    raise ValueError(f"Empty combined AnnData object: {adata.n_obs} cells, {adata.n_vars} genes.")

# Store log-normalized expression separately; restore counts to X for HVG/PCA.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()
adata.X = adata.layers["counts"].copy()

# Pearson-residual HVGs are computed on raw counts.
sc.experimental.pp.highly_variable_genes(
    adata, flavor="pearson_residuals", n_top_genes=min(3000, adata.n_vars),
    batch_key="sample",
)
hv_mask = adata.var["highly_variable"].to_numpy()
if int(hv_mask.sum()) < 3:
    raise ValueError("Fewer than 3 highly variable genes were selected.")
hv = adata[:, hv_mask].copy()
sc.experimental.pp.normalize_pearson_residuals(hv, theta=100)

n_pcs = min(30, hv.n_vars - 1, hv.n_obs - 1)
if n_pcs < 2:
    raise ValueError(f"Not enough cells/genes for PCA (cells={hv.n_obs}, genes={hv.n_vars}).")
sc.pp.pca(hv, n_comps=n_pcs, random_state=SEED)
adata.obsm["X_pca"] = hv.obsm["X_pca"]

ho = harmonypy.run_harmony(
    adata.obsm["X_pca"].astype(np.float64),
    adata.obs,
    "sample",
)
z_corr = np.asarray(ho.Z_corr)
adata.obsm["X_pca_harmony"] = z_corr.T if z_corr.shape[0] != adata.n_obs else z_corr

n_pcs_neighbors = min(n_pcs, adata.obsm["X_pca_harmony"].shape[1])
sc.pp.neighbors(
    adata, use_rep="X_pca_harmony", n_pcs=n_pcs_neighbors,
    n_neighbors=min(40, max(2, adata.n_obs - 1)), random_state=SEED,
)
sc.tl.umap(adata, min_dist=0.5, spread=1.5, random_state=SEED)


# ----------------------------------------------------------------------------
# 3. RESOLUTION SEARCH (TARGET 22 CLUSTERS; NOT CELL-TYPE ANNOTATION)
# ----------------------------------------------------------------------------
target_clusters = 22
resolution_grid = np.round(np.arange(0.2, 3.01, 0.1), 2)
best = None

for resolution in resolution_grid:
    sc.tl.leiden(
        adata, resolution=float(resolution), key_added="clusters",
        flavor="igraph", directed=False, random_state=SEED,
    )
    n_clusters = adata.obs["clusters"].nunique()
    print(f"Resolution {resolution:.2f} -> {n_clusters} clusters")
    candidate = (abs(n_clusters - target_clusters), float(resolution), n_clusters)
    if best is None or candidate < best:
        best = candidate
    if n_clusters == target_clusters:
        best = (0, float(resolution), n_clusters)
        break

if best is None:
    raise RuntimeError("Leiden resolution search did not run.")

# Recompute at the best resolution and validate the cluster count.
best_resolution = best[1]
sc.tl.leiden(
    adata, resolution=best_resolution, key_added="clusters",
    flavor="igraph", directed=False, random_state=SEED,
)
n_clusters = adata.obs["clusters"].nunique()
print(f"Selected resolution {best_resolution:.2f}: {n_clusters} clusters")
if n_clusters != target_clusters:
    warnings.warn(
        f"Could not obtain exactly {target_clusters} clusters; obtained "
        f"{n_clusters}. Cluster IDs will be retained without arbitrary "
        "cell-type assignment. Adjust the resolution grid if exactly 22 "
        "clusters is essential.",
        RuntimeWarning,
    )

# No arbitrary cluster-to-cell-type mapping. Use "clusters" for plotting until
# validated marker-based/reference-based annotations have been supplied.
adata.obs["cell_type"] = pd.Categorical(
    ["Unannotated"] * adata.n_obs,
    categories=["Unannotated"],
)

sc.pl.umap(
    adata, color=["clusters", "cell_type"], ncols=1, show=False,
)
plt.savefig(OUT / "umap_clusters_unannotated.png", bbox_inches="tight", dpi=300)
plt.close()

sc.pl.umap(
    adata,
    color="gravity",
    title="PBMCs: 1G vs. Simulated Microgravity (uG)",
    palette={"1G": "#377EB8", "uG": "#E41A1C"},
    frameon=False,
    legend_loc="right margin",
    show=False,
)
plt.savefig(OUT / "umap_gravity_1G_vs_uG.png", dpi=300, bbox_inches="tight")
plt.close()

for stim_condition in ["unstim", "stim"]:
    adata_subset = adata[adata.obs["stim"] == stim_condition].copy()
    if adata_subset.n_obs == 0:
        warnings.warn(f"No cells for stim={stim_condition}; skipping UMAP.")
        continue
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
        dpi=300, bbox_inches="tight",
    )
    plt.close()

sc.pl.umap(
    adata, color="sample", title="PBMC UMAP by Sample",
    frameon=False, show=False,
)
plt.savefig(OUT / "umap_by_sample.png", dpi=300, bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------------
# 4. EXPLORATORY DIFFERENTIAL EXPRESSION & VOLCANO PLOT
# ----------------------------------------------------------------------------
# This is an exploratory cell-level test, not replicate-aware inference.
# Restrict to unstimulated cells to match the volcano plot title.
adata.X = adata.layers["lognorm"].copy()
de_adata = adata[adata.obs["stim"].astype(str) == "unstim"].copy()

if set(de_adata.obs["gravity"].astype(str).unique()) >= {"1G", "uG"}:
    sc.tl.rank_genes_groups(
        de_adata, groupby="gravity", groups=["uG"], reference="1G",
        method="wilcoxon",
    )
    deg_df = sc.get.rank_genes_groups_df(de_adata, group="uG")
    deg_df["-log10(pvals_adj)"] = -np.log10(
        deg_df["pvals_adj"].clip(lower=1e-300)
    )

    pval_thresh = 0.05
    logfc_thresh = 0.25
    sig_both = (
        (deg_df["pvals_adj"] < pval_thresh)
        & (deg_df["logfoldchanges"].abs() > logfc_thresh)
    )
    sig_pval_only = (
        (deg_df["pvals_adj"] < pval_thresh)
        & (deg_df["logfoldchanges"].abs() <= logfc_thresh)
    )
    not_sig = ~(sig_both | sig_pval_only)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        deg_df.loc[not_sig, "logfoldchanges"],
        deg_df.loc[not_sig, "-log10(pvals_adj)"],
        s=10, c="lightgrey", alpha=0.6, edgecolors="none", label="NS",
    )
    ax.scatter(
        deg_df.loc[sig_pval_only, "logfoldchanges"],
        deg_df.loc[sig_pval_only, "-log10(pvals_adj)"],
        s=10, c="cornflowerblue", alpha=0.8, edgecolors="none", label="Adjusted p-value",
    )
    ax.scatter(
        deg_df.loc[sig_both, "logfoldchanges"],
        deg_df.loc[sig_both, "-log10(pvals_adj)"],
        s=15, c="firebrick", alpha=0.9, edgecolors="none",
        label="Adjusted p-value + |log2FC| threshold",
    )

    target_genes = deg_df[deg_df["names"].isin(DEGS)]
    texts = [
        ax.text(row["logfoldchanges"], row["-log10(pvals_adj)"], row["names"], fontsize=8)
        for _, row in target_genes.iterrows()
    ]
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.axhline(y=-np.log10(pval_thresh), color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Log fold change (Scanpy rank_genes_groups output)")
    ax.set_ylabel("-log10 adjusted p-value")
    ax.set_title("uG vs 1G\nunstimulated PBMC", loc="left", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, title="Significance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "volcano_plot_uG_vs_1G_unstim.png", bbox_inches="tight", dpi=300)
    plt.close()
    deg_df.to_csv(OUT / "differential_expression_uG_vs_1G_unstim.csv", index=False)
else:
    warnings.warn(
        "Skipping DE/volcano plot: unstimulated subset does not contain both "
        "gravity groups '1G' and 'uG'.",
        RuntimeWarning,
    )


# ----------------------------------------------------------------------------
# 5. DOT PLOT (ONLY GENES PRESENT IN THE DATA)
# ----------------------------------------------------------------------------
top_genes = [
    "MMP9", "RBM3", "HNRNPA1", "STAT1", "CIRBP",
    "CCL2", "SLC7A11", "SERF2", "NFKBIA", "THBS1",
]
present_genes = [gene for gene in top_genes if gene in adata.var_names]
missing_genes = sorted(set(top_genes) - set(present_genes))
if missing_genes:
    warnings.warn(f"Dot-plot genes absent from adata.var_names: {missing_genes}")
if present_genes:
    sc.pl.dotplot(
        adata, var_names=present_genes, groupby="clusters",
        standard_scale="var", cmap="RdYlBu_r", show=False,
    )
    plt.savefig(OUT / "dotplot_genes_vs_clusters.png", bbox_inches="tight", dpi=300)
    plt.close()
else:
    warnings.warn("Skipping dot plot: none of the requested genes are present.")


# ----------------------------------------------------------------------------
# 6. SENDER-RECEIVER LIGAND-RECEPTOR ANALYSIS (EXPLORATORY)
# ----------------------------------------------------------------------------
# IMPORTANT: this section intentionally stops unless valid cell-type annotations
# are supplied. Leiden cluster IDs must not be relabeled arbitrarily.
CELL_TYPE_COL = "cell_type"
GRAVITY_COL = "gravity"
STIM_COL = "stim"
SAMPLE_COL = "sample"

SENDER_LABELS = {
    "D_B_cells": ["B memory", "B intermediate", "B naive", "Plasmablast"],
    "E_DCs": ["cDC2", "pDC"],
    "F_Monocytes": ["CD14 Mono", "CD16 Mono"],
}
T_CELL_LABELS = [
    "CD4 Naive", "CD4 TCM", "CD8 TEM", "CD4 TEM", "CD8 Naive",
    "dnT", "CD8 TCM", "gdT", "Treg", "CD4 CTL", "MAIT",
]
STIM_CONDITIONS = ["unstim", "stim"]

required_cols = [CELL_TYPE_COL, GRAVITY_COL, STIM_COL, SAMPLE_COL]
missing_cols = [col for col in required_cols if col not in adata.obs.columns]
if missing_cols:
    raise KeyError(f"Missing metadata columns in adata.obs: {missing_cols}")

available_types = set(adata.obs[CELL_TYPE_COL].astype(str).unique())
requested_types = set(T_CELL_LABELS)
for labels in SENDER_LABELS.values():
    requested_types.update(labels)

# This guard is expected to trigger until cell_type is populated with validated
# annotations. The preceding QC, clustering, UMAP, DE and dot plot remain saved.
if available_types == {"Unannotated"} or not requested_types.issubset(available_types):
    missing_types = sorted(requested_types - available_types)
    message = (
        "Skipping LIANA+ analysis because validated cell-type annotations have "
        "not been supplied. Do not interpret cluster IDs as cell types. "
        f"Missing requested labels: {missing_types}"
    )
    warnings.warn(message, RuntimeWarning)
else:
    if "lognorm" not in adata.layers:
        raise KeyError("adata.layers['lognorm'] is required for LIANA+.")

    def run_liana_by_condition(adata_input, stim_condition, gravity_condition):
        """Infer candidate ligand-receptor interactions within one condition."""
        labels = list(T_CELL_LABELS) + [
            label for values in SENDER_LABELS.values() for label in values
        ]
        mask = (
            (adata_input.obs[STIM_COL].astype(str) == stim_condition)
            & (adata_input.obs[GRAVITY_COL].astype(str) == gravity_condition)
            & adata_input.obs[CELL_TYPE_COL].astype(str).isin(labels)
        )
        subset = adata_input[mask].copy()
        subset.X = subset.layers["lognorm"].copy()
        if subset.n_obs == 0:
            raise ValueError(
                f"No cells found for stim={stim_condition}, gravity={gravity_condition}."
            )
        li.mt.rank_aggregate(
            subset, groupby=CELL_TYPE_COL, use_raw=False,
            expr_prop=0.1, min_cells=10, verbose=False,
        )
        result = subset.uns.get("liana_res")
        if result is None:
            raise RuntimeError(
                "LIANA+ did not return 'liana_res'. Check the installed LIANA+ "
                "version and rank_aggregate output documentation."
            )
        result = result.copy()
        result["stim_condition"] = stim_condition
        result["gravity_condition"] = gravity_condition
        return result

    def filter_sender_receiver(result, sender_labels):
        """Keep sender -> T-cell receiver interactions after schema validation."""
        required = {"source", "target", "ligand_complex", "receptor_complex"}
        absent = required - set(result.columns)
        if absent:
            raise KeyError(
                "Unexpected LIANA+ result schema; missing columns: "
                f"{sorted(absent)}. Available columns: {list(result.columns)}"
            )
        return result.loc[
            result["source"].isin(sender_labels)
            & result["target"].isin(T_CELL_LABELS)
        ].copy()

    liana_results = {}
    for stim_condition in STIM_CONDITIONS:
        for gravity_condition in ["1G", "uG"]:
            key = f"{stim_condition}_{gravity_condition}"
            print(f"Running LIANA+: {key}")
            liana_results[key] = run_liana_by_condition(
                adata, stim_condition, gravity_condition,
            )

    for stim_condition in STIM_CONDITIONS:
        res_1g = liana_results[f"{stim_condition}_1G"].copy()
        res_ug = liana_results[f"{stim_condition}_uG"].copy()

        for panel, sender_labels in SENDER_LABELS.items():
            panel_1g = filter_sender_receiver(res_1g, sender_labels)
            panel_ug = filter_sender_receiver(res_ug, sender_labels)
            panel_1g.to_csv(OUT / f"{panel}_{stim_condition}_1G_interactions.csv", index=False)
            panel_ug.to_csv(OUT / f"{panel}_{stim_condition}_uG_interactions.csv", index=False)

            score_candidates = [
                "magnitude_rank", "specificity_rank", "lrscore",
                "natmi_score", "aggregate_rank",
            ]
            score_col = next(
                (col for col in score_candidates
                 if col in panel_1g.columns and col in panel_ug.columns),
                None,
            )
            if score_col is None:
                print(
                    f"{panel} ({stim_condition}): no common score column; "
                    "saved interaction tables without score deltas."
                )
                continue

            keys = ["source", "target", "ligand_complex", "receptor_complex"]
            comparison = panel_ug[keys + [score_col]].merge(
                panel_1g[keys + [score_col]],
                on=keys, how="outer", suffixes=("_uG", "_1G"),
            )
            if "rank" in score_col:
                comparison["delta_uG_minus_1G"] = (
                    comparison[f"{score_col}_1G"] - comparison[f"{score_col}_uG"]
                )
                interpretation = "Positive = higher-ranked in uG"
            else:
                comparison["delta_uG_minus_1G"] = (
                    comparison[f"{score_col}_uG"] - comparison[f"{score_col}_1G"]
                )
                interpretation = "Positive = larger score in uG"
            comparison["delta_interpretation"] = interpretation
            comparison = comparison.sort_values(
                "delta_uG_minus_1G", ascending=False, na_position="last",
            )
            comparison.to_csv(
                OUT / f"{panel}_{stim_condition}_uG_vs_1G_comparison.csv",
                index=False,
            )

            top = comparison.dropna(subset=["delta_uG_minus_1G"]).head(20).copy()
            if top.empty:
                continue
            top["interaction"] = (
                top["source"].astype(str) + ": "
                + top["ligand_complex"].astype(str) + " → "
                + top["receptor_complex"].astype(str) + " → "
                + top["target"].astype(str)
            )
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(
                data=top, x="delta_uG_minus_1G", y="interaction",
                ax=ax, color="steelblue",
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(
                f"{panel}: Sender → T-cell interactions\n"
                f"{stim_condition}, uG vs 1G ({score_col})"
            )
            ax.set_xlabel("Descriptive score difference (positive favors uG)")
            ax.set_ylabel("")
            plt.tight_layout()
            plt.savefig(
                OUT / f"{panel}_{stim_condition}_uG_vs_1G.png",
                dpi=300, bbox_inches="tight",
            )
            plt.close()

# Save the processed object and cluster/metadata table for reproducibility.
adata.write_h5ad(OUT / "pbmc_gse218937_processed.h5ad", compression="gzip")
adata.obs.to_csv(OUT / "pbmc_cell_metadata.csv")
print(f"Pipeline outputs saved under: {OUT.resolve()}")
