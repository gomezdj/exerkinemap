#!/usr/bin/env python
"""
CORENTO: pathway activity and multi-sample CARNIVAL network inference.

Inputs are outputs from pbmc_scrnaseq_pipeline.py in ./pbmc_out:
  de_1G.csv, de_uG.csv, de_unstim.csv, de_stim.csv
  pbmc_integrated_clusters.h5ad (optional)

Outputs are written to pbmc_out/pathways/.

This script infers pathway/TF activity and candidate signaling networks. Scores
are computational inferences, not direct measurements of protein activity.
The difference between two ULM scores is exploratory and is NOT a formal
gravity-by-stimulation interaction test.

Install (check API compatibility for your installed package versions):
  pip install "gseapy" "decoupler>=2" omnipath corneto cvxpy networkx scanpy \
      scipy pandas numpy matplotlib

CORNETO graph convention follows its documented CARNIVAL examples:
  Graph.from_tuples([("source", sign, "target"), ...])
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats

import gseapy as gp
import decoupler as dc
import omnipath as op
import corneto as cn
from corneto.graph import Graph

try:
    from corneto.methods import CarnivalFlow
except ImportError:
    from corneto.methods.future.carnival import CarnivalFlow

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
OUT = Path("pbmc_out").resolve()
RES = OUT / "pathways"

P_CUTOFF = 0.05
LFC_CUTOFF = 0.1
TOP_TF = 20
LAMBDA = 1e-3
SOLVER = "scipy"
MAX_ROOTS = 150

# Supply known perturbation/input nodes when experimentally justified.
# Example: INPUTS = {"TLR4": 1}
INPUTS: dict[str, int] = {}

CONTRASTS = {
    "1G_stim_vs_unstim": "de_1G.csv",
    "uG_stim_vs_unstim": "de_uG.csv",
    "uG_vs_1G_unstim": "de_unstim.csv",
    "uG_vs_1G_stim": "de_stim.csv",
}
NET_CONTRASTS = ["1G_stim_vs_unstim", "uG_stim_vs_unstim"]
REQUIRED_DE_COLUMNS = {"p_val", "avg_log2FC", "p_val_adj"}
# -----------------------------------------------------------------------------
# PRIORITY PATHWAYS: SIRTUIN SIGNALING AND OXPHOS
# -----------------------------------------------------------------------------
PRIORITY_PATHWAYS = {
    "Sirtuin_Signaling": [
        "sirtuin",
        "sirtuins",
        "sirtuin signaling",
        "regulation of sirtuin",
    ],
    "Oxidative_Phosphorylation": [
        "oxidative phosphorylation",
        "respiratory electron transport",
        "electron transport chain",
        "mitochondrial ATP synthesis",
    ],
}

def validate_inputs() -> dict[str, pd.DataFrame]:
    """Validate expected files, columns, gene identifiers, and numeric values."""
    if not OUT.is_dir():
        raise FileNotFoundError(f"Pipeline output directory not found: {OUT}")

    loaded = {}
    for contrast, filename in CONTRASTS.items():
        path = OUT / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing DE table for {contrast}: {path}")

        df = pd.read_csv(path, index_col=0)
        if df.empty:
            raise ValueError(f"DE table is empty for {contrast}: {path}")

        missing = REQUIRED_DE_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{filename} is missing required columns: {sorted(missing)}"
            )
        if df.index.has_duplicates:
            raise ValueError(f"{filename} has duplicate gene identifiers.")
        if df.index.isna().any():
            raise ValueError(f"{filename} contains missing gene identifiers.")

        for col in REQUIRED_DE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[["p_val", "avg_log2FC", "p_val_adj"]].isna().any().any():
            raise ValueError(f"{filename} has nonnumeric/missing required values.")
        if ((df["p_val"] < 0) | (df["p_val"] > 1)).any():
            raise ValueError(f"{filename}: p_val must be in [0, 1].")
        if ((df["p_val_adj"] < 0) | (df["p_val_adj"] > 1)).any():
            raise ValueError(f"{filename}: p_val_adj must be in [0, 1].")

        # Keep IDs as supplied; uppercasing is not an Ensembl-to-symbol conversion.
        loaded[contrast] = df

    return loaded


def pick_library(prefix: str) -> str:
    """Select a matching Enrichr library, failing clearly if unavailable."""
    libraries = gp.get_library_name(organism="Human")
    names = [name for name in libraries if name.startswith(prefix)]
    if not names:
        raise ValueError(
            f"No Enrichr library starts with {prefix!r}. "
            "Check GSEApy connectivity/library availability."
        )
    return sorted(names)[-1]


def run_ora(genes, background, contrast: str, direction: str,
            libraries: list[str]) -> pd.DataFrame:
    """Run ORA; background should be the tested, identifier-compatible gene universe."""
    genes = sorted({str(g).strip().upper() for g in genes if pd.notna(g)})
    bg = sorted({str(g).strip().upper() for g in background if pd.notna(g)})
    genes = [g for g in genes if g and g in set(bg)]
    if len(genes) < 5 or len(bg) < len(genes):
        return pd.DataFrame()

    frames = []
    for library in libraries:
        enr = gp.enrich(
            gene_list=genes,
            gene_sets=library,
            background=bg,
            outdir=None,
            no_plot=True,
        )
        result = getattr(enr, "results", None)
        if result is None or result.empty:
            continue
        result = result.copy()
        result["library"] = library
        frames.append(result)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["contrast"] = contrast
    out["direction"] = direction
    return out


def empty_ora_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Term", "Adjusted P-value", "contrast", "direction", "library"]
    )

def extract_priority_pathways(ora: pd.DataFrame) -> pd.DataFrame:
    """Extract Sirtuin and OXPHOS-related terms from enrichment results."""
    if ora.empty or "Term" not in ora.columns:
        return pd.DataFrame(
            columns=list(ora.columns) + ["priority_pathway"]
        )

    terms = ora["Term"].astype(str).str.lower()
    frames = []

    for pathway, keywords in PRIORITY_PATHWAYS.items():
        pattern = "|".join(
            __import__("re").escape(keyword.lower())
            for keyword in keywords
        )
        selected = ora.loc[terms.str.contains(pattern, regex=True)].copy()

        if not selected.empty:
            selected["priority_pathway"] = pathway
            frames.append(selected)

    if not frames:
        return pd.DataFrame(
            columns=list(ora.columns) + ["priority_pathway"]
        )

    return pd.concat(frames, ignore_index=True).drop_duplicates()

def plot_ora(ora: pd.DataFrame, contrast: str, direction: str, n: int = 12):
    if ora.empty or not {"contrast", "direction", "Adjusted P-value", "Term"} <= set(ora.columns):
        return
    d = ora[(ora["contrast"] == contrast) & (ora["direction"] == direction)].copy()
    if d.empty:
        return
    d["Adjusted P-value"] = pd.to_numeric(d["Adjusted P-value"], errors="coerce")
    d = d.dropna(subset=["Adjusted P-value"]).nsmallest(n, "Adjusted P-value")
    if d.empty:
        return
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 1.5))
    ax.barh(
        d["Term"].astype(str).str.slice(0, 60),
        -np.log10(d["Adjusted P-value"].clip(lower=1e-300)),
        color="tab:red" if direction == "up" else "tab:blue",
    )
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(f"{contrast} ({direction})")
    fig.savefig(RES / f"ora_{contrast}_{direction}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def signed_z(df: pd.DataFrame) -> pd.Series:
    """Signed normal score from nominal p-value and logFC; not a model z-statistic."""
    p = np.clip(df["p_val"].to_numpy(dtype=float), 1e-300, 1.0)
    direction = np.sign(df["avg_log2FC"].to_numpy(dtype=float))
    values = stats.norm.isf(p / 2.0) * direction
    values[~np.isfinite(values)] = 0.0
    return pd.Series(values, index=df.index)


def tidy_activity(scores: pd.DataFrame, padj: pd.DataFrame, kind: str) -> pd.DataFrame:
    scores = scores.rename_axis(index="contrast", columns="source")
    padj = padj.rename_axis(index="contrast", columns="source")
    out = pd.concat(
        [scores.stack().rename("score"), padj.stack().rename("padj")], axis=1
    )
    return out.reset_index().assign(kind=kind)


def barplot_activity(act: pd.DataFrame, contrast: str, kind: str, n: int = 20):
    d = act[(act["contrast"] == contrast) & (act["kind"] == kind)].copy()
    if d.empty:
        return
    d = d.reindex(d["score"].abs().sort_values(ascending=False).index).head(n)
    d = d.sort_values("score")
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(d) + 1.5))
    ax.barh(
        d["source"],
        d["score"],
        color=np.where(d["score"] > 0, "tab:red", "tab:blue"),
    )
    ax.set_xlabel("ULM score (inferred positive/negative activity)")
    ax.set_title(f"{kind}: {contrast}")
    fig.savefig(
        RES / f"activity_{kind}_{contrast}.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def progeny_on_umap(h5ad_path: Path, progeny_net):
    """Compute per-cell PROGENy scores after checking AnnData inputs."""
    if not h5ad_path.is_file():
        print(f"Skipping per-cell PROGENy: file not found: {h5ad_path}")
        return

    adata = sc.read_h5ad(h5ad_path)
    if "X_umap" not in adata.obsm:
        raise KeyError("AnnData is missing obsm['X_umap'].")
    missing_obs = {"group", "clusters"} - set(adata.obs.columns)
    if missing_obs:
        raise KeyError(f"AnnData is missing obs columns: {sorted(missing_obs)}")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("AnnData has zero cells or zero genes.")

    # decoupler expects a suitable expression matrix; confirm its scale in the source
    # pipeline (e.g., normalized/log-transformed expression) before interpreting scores.
    dc.mt.ulm(adata, progeny_net, tmin=5)
    pa = dc.pp.get_obsm(adata, key="score_ulm")
    pa.obsm["X_umap"] = adata.obsm["X_umap"]

    sc.pl.umap(
        pa,
        color=list(pa.var_names),
        cmap="RdBu_r",
        vcenter=0,
        ncols=4,
        show=False,
    )
    plt.savefig(RES / "umap_progeny.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    for key in ("group", "clusters"):
        m = pa.to_df().join(adata.obs[[key]]).groupby(
            key, observed=True
        ).mean(numeric_only=True)
        if m.empty:
            continue
        vmax = float(np.nanmax(np.abs(m.to_numpy())))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        fig, ax = plt.subplots(figsize=(0.6 * m.shape[1] + 2, 0.35 * m.shape[0] + 2))
        im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(m.shape[1]), m.columns, rotation=90)
        ax.set_yticks(range(m.shape[0]), m.index)
        fig.colorbar(im, ax=ax, label="mean inferred ULM score")
        fig.savefig(RES / f"progeny_by_{key}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def build_pkn() -> pd.DataFrame:
    """Retrieve signed, directed OmniPath interactions as source/sign/target rows."""
    ia = op.interactions.OmniPath.get(genesymbols=True)
    required = {
        "source_genesymbol", "target_genesymbol",
        "is_directed", "is_stimulation", "is_inhibition",
    }
    missing = required - set(ia.columns)
    if missing:
        raise ValueError(
            "OmniPath response schema differs from expected API; missing "
            f"columns: {sorted(missing)}. Available: {ia.columns.tolist()}"
        )

    ia = ia.copy()
    for col in ("is_directed", "is_stimulation", "is_inhibition"):
        # Avoid silently converting missing flags into True.
        ia[col] = ia[col].fillna(False).astype(bool)

    ia = ia[
        ia["is_directed"]
        & (ia["is_stimulation"] ^ ia["is_inhibition"])
    ].copy()

    edges = pd.DataFrame({
        "source": ia["source_genesymbol"],
        "target": ia["target_genesymbol"],
        "sign": np.where(ia["is_stimulation"], 1, -1),
    }).dropna()
    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()
    edges = edges[
        (edges["source"] != "")
        & (edges["target"] != "")
        & (edges["source"] != edges["target"])
    ].drop_duplicates()
    if edges.empty:
        raise RuntimeError("OmniPath filtering produced an empty signed PKN.")
    return edges.reset_index(drop=True)


def prune(edges: pd.DataFrame, outputs, inputs=None) -> pd.DataFrame:
    """Keep nodes that reach an output and, if supplied, are reachable from inputs."""
    graph = nx.DiGraph()
    graph.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    outputs = [o for o in outputs if o in graph]
    keep = set(outputs)
    for output in outputs:
        keep |= nx.ancestors(graph, output)
    if inputs:
        reachable = set()
        for node in inputs:
            if node in graph:
                reachable.add(node)
                reachable |= nx.descendants(graph, node)
        keep &= reachable
    return edges[
        edges["source"].isin(keep) & edges["target"].isin(keep)
    ].reset_index(drop=True)


def plot_network(edges_df: pd.DataFrame, nodes_df: pd.DataFrame,
                 sample: str, path: Path):
    """Plot selected edge assignments; edge assignment sign is not PKN interaction sign."""
    if edges_df.empty or sample not in edges_df:
        return
    e = edges_df[edges_df[sample] != 0]
    if e.empty:
        return
    graph = nx.DiGraph()
    for _, row in e.iterrows():
        graph.add_edge(row["source"], row["target"], assignment=row[sample])

    node_activity = nodes_df.set_index("vertex")[sample]
    colors = [
        {1: "#d6604d", -1: "#4393c3"}.get(int(node_activity.get(n, 0)), "#cccccc")
        for n in graph.nodes
    ]
    try:
        pos = nx.kamada_kawai_layout(graph)
    except Exception:
        pos = nx.spring_layout(graph, seed=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=500, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)
    positive = [(u, v) for u, v, d in graph.edges(data=True) if d["assignment"] > 0]
    negative = [(u, v) for u, v, d in graph.edges(data=True) if d["assignment"] < 0]
    nx.draw_networkx_edges(graph, pos, edgelist=positive, edge_color="tab:red", arrows=True, ax=ax)
    nx.draw_networkx_edges(
        graph, pos, edgelist=negative, edge_color="tab:blue",
        style="dashed", arrows=True, ax=ax
    )
    ax.set_title(
        f"{sample}: edge assignment (red = positive, blue dashed = negative); "
        "not the prior interaction sign"
    )
    ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    RES.mkdir(parents=True, exist_ok=True)
    de = validate_inputs()
    print("Validated DE contrasts:", {k: v.shape for k, v in de.items()})

    # 1. ORA of up/down DEGs. Background = genes tested in each contrast.
    libs = [pick_library(prefix) for prefix in ("MSigDB_Hallmark", "Reactome", "KEGG")]
    print("Enrichr libraries:", libs)

    ora_frames = []
    for name, df in de.items():
        sig = df[df["p_val_adj"] < P_CUTOFF]
        gene_ids = df.index.astype(str)
        sets = {
            "up": sig.index[sig["avg_log2FC"] > LFC_CUTOFF],
            "down": sig.index[sig["avg_log2FC"] < -LFC_CUTOFF],
        }
        for direction, genes in sets.items():
            result = run_ora(genes, gene_ids, name, direction, libs)
            if not result.empty:
                ora_frames.append(result)

    ora = pd.concat(ora_frames, ignore_index=True) if ora_frames else empty_ora_frame()
    ora.to_csv(RES / "ora_degs.csv", index=False)
    for name in de:
        for direction in ("up", "down"):
            plot_ora(ora, name, direction)

    # 2. Contrast-by-gene signed evidence and decoupler ULM footprints.
    # Best practice: use complete, unfiltered DE tables. Missing genes set to zero
    # here are not evidence of no effect; they may have been filtered upstream.
    stat = pd.DataFrame({k: signed_z(v) for k, v in de.items()}).T.fillna(0.0)
    if not np.isfinite(stat.to_numpy()).all():
        raise ValueError("Signed-statistic matrix contains non-finite values.")

    nets = {
        "progeny": dc.op.progeny(organism="human", top=500),
        "collectri": dc.op.collectri(organism="human"),
        "hallmark": dc.op.hallmark(organism="human"),
    }
    act_frames = []
    for kind, net in nets.items():
        scores, padj = dc.mt.ulm(stat, net, tmin=5)
        act_frames.append(tidy_activity(scores, padj, kind))
    act = pd.concat(act_frames, ignore_index=True) if act_frames else pd.DataFrame()
    if act.empty:
        raise RuntimeError("decoupler returned no activity scores.")
    act.to_csv(RES / "activity_ulm.csv", index=False)

    for kind in nets:
        for name in de:
            barplot_activity(act, name, kind)

    # Exploratory score difference only; not a formal interaction test.
    tf_wide = act[act["kind"] == "collectri"].pivot(
        index="source", columns="contrast", values="score"
    )
    needed = {"uG_stim_vs_unstim", "1G_stim_vs_unstim"}
    if needed <= set(tf_wide.columns):
        tf_wide["delta_uG_minus_1G_exploratory"] = (
            tf_wide["uG_stim_vs_unstim"] - tf_wide["1G_stim_vs_unstim"]
        )
        tf_wide.sort_values("delta_uG_minus_1G_exploratory").to_csv(
            RES / "tf_delta_uG_vs_1G_exploratory.csv"
        )
    else:
        print("Skipping TF delta: one or more primary contrasts are missing.")

    # 3. Optional per-cell PROGENy.
    try:
        progeny_on_umap(OUT / "pbmc_integrated_clusters.h5ad", nets["progeny"])
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"Skipping per-cell PROGENy due to input/API issue: {exc}")

    # 4. CARNIVAL multi-sample network inference.
    pkn = build_pkn()
    pkn_nodes = set(pkn["source"]) | set(pkn["target"])
    print(f"OmniPath PKN: {len(pkn):,} signed edges, {len(pkn_nodes):,} nodes")

    tf_tab = act[act["kind"] == "collectri"].copy()
    sig = tf_tab[
        (tf_tab["padj"] < 0.05) & tf_tab["contrast"].isin(NET_CONTRASTS)
    ].copy()
    if sig.empty:
        raise RuntimeError("No significant CollecTRI TF scores for network contrasts.")
    sig["abs_score"] = sig["score"].abs()
    top = sig.sort_values("abs_score", ascending=False).groupby("contrast").head(TOP_TF)
    tf_set = sorted(set(top["source"]) & pkn_nodes)
    if not tf_set:
        raise RuntimeError("No significant TFs overlap the OmniPath PKN.")

    tf_scores = tf_tab.pivot(index="source", columns="contrast", values="score")
    missing_contrasts = set(NET_CONTRASTS) - set(tf_scores.columns)
    if missing_contrasts:
        raise ValueError(f"Missing network contrasts: {sorted(missing_contrasts)}")
    tf_scores = tf_scores.reindex(index=tf_set, columns=NET_CONTRASTS)

    # Select only TFs with finite evidence in both samples.
    valid = np.isfinite(tf_scores.to_numpy()).all(axis=1)
    tf_scores = tf_scores.loc[valid]
    tf_set = list(tf_scores.index)
    if not tf_set:
        raise RuntimeError("No TFs have finite scores in both network contrasts.")

    scale = float(tf_scores.abs().to_numpy().max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("TF score normalization is undefined (all scores zero/non-finite).")
    tf_vals = tf_scores / scale

    sub = prune(pkn, tf_set, list(INPUTS) or None)
    inverse = not bool(INPUTS)
    if inverse:
        targets = set(sub["target"])
        roots = [
            node for node in set(sub["source"]) - targets
            if node not in tf_set
        ]
        if not roots:
            raise RuntimeError("Inverse mode found no candidate roots after pruning.")
        if len(roots) > MAX_ROOTS:
            outdegree = sub["source"].value_counts()
            roots = list(outdegree.reindex(roots).fillna(0).sort_values(ascending=False).index[:MAX_ROOTS])
        sub = prune(sub, tf_set, roots)
        rows = list(zip(sub["source"], sub["sign"].astype(int), sub["target"]))
        for root in roots:
            rows.extend([
                ("PERTURBATION", 1, root),
                ("PERTURBATION", -1, root),
            ])
    else:
        rows = list(zip(sub["source"], sub["sign"].astype(int), sub["target"]))

    print(
        f"Pruned PKN: {len(sub):,} edges, "
        f"{len(set(sub['source']) | set(sub['target'])):,} nodes"
    )

    # Drop output TFs with no incoming edge in the pruned PKN.
    reachable_tfs = set(sub["target"])
    dropped = [tf for tf in tf_set if tf not in reachable_tfs]
    if dropped:
        print("Dropping TFs without incoming edges in pruned PKN:", dropped)
    tf_set = [tf for tf in tf_set if tf in reachable_tfs]
    if not tf_set:
        raise RuntimeError("No TF outputs are reachable in the pruned PKN.")
    tf_vals = tf_vals.loc[tf_set]

    # CORNETO documented signed-edge tuple format: (source, sign, target).
    graph = Graph.from_tuples(rows)
    samples = {}
    for contrast in NET_CONTRASTS:
        sample_data = {
            tf: {
                "value": float(value),
                "mapping": "vertex",
                "role": "output",
            }
            for tf, value in tf_vals[contrast].items()
        }
        for node, value in INPUTS.items():
            sample_data[node] = {
                "value": float(value), "mapping": "vertex", "role": "input"
            }
        if inverse:
            sample_data["PERTURBATION"] = {
                "value": 1, "mapping": "vertex", "role": "input"
            }
        samples[contrast] = sample_data

    data = cn.Data.from_cdict(samples)
    carnival = CarnivalFlow(lambda_reg=LAMBDA)
    problem = carnival.build(graph, data)
    problem.solve(solver=SOLVER, verbosity=1)
    for objective in problem.objectives:
        print("objective:", objective.value)

    # CORNETO's current CARNIVAL examples expose per-sample edge_value and
    # vertex_value. These are model assignments; do not confuse edge assignment
    # sign with the prior-knowledge interaction sign.
    processed_graph = carnival.processed_graph
    if not hasattr(problem.expr, "edge_value") or not hasattr(problem.expr, "vertex_value"):
        raise AttributeError(
            "Installed CORNETO API does not expose expr.edge_value and "
            "expr.vertex_value as expected; inspect its CARNIVAL documentation."
        )
    edge_values = np.asarray(problem.expr.edge_value.value)
    vertex_values = np.asarray(problem.expr.vertex_value.value)
    if edge_values.size == 0 or vertex_values.size == 0:
        raise RuntimeError("CORNETO returned empty solution variables.")
    edge_values = edge_values.reshape(len(edge_values), -1)
    vertex_values = vertex_values.reshape(len(vertex_values), -1)
    if edge_values.shape[1] != len(NET_CONTRASTS):
        raise ValueError(
            f"Unexpected edge solution shape {edge_values.shape}; "
            f"expected {len(NET_CONTRASTS)} sample columns."
        )
    if vertex_values.shape[1] != len(NET_CONTRASTS):
        raise ValueError(
            f"Unexpected vertex solution shape {vertex_values.shape}; "
            f"expected {len(NET_CONTRASTS)} sample columns."
        )

    def node_name(value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return next(iter(value)) if value else None
        return value

    edge_rows = [
        (node_name(source), node_name(target))
        for source, target in processed_graph.E
    ]
    edges_df = pd.DataFrame(edge_rows, columns=["source", "target"])
    for i, contrast in enumerate(NET_CONTRASTS):
        edges_df[contrast] = np.rint(edge_values[:, i]).astype(int)
    edges_df = edges_df.dropna(subset=["source", "target"])
    edges_df = edges_df[(edges_df[NET_CONTRASTS] != 0).any(axis=1)]

    nodes_df = pd.DataFrame({"vertex": [node_name(v) for v in processed_graph.V]})
    for i, contrast in enumerate(NET_CONTRASTS):
        nodes_df[contrast] = np.rint(vertex_values[:, i]).astype(int)
    nodes_df = nodes_df[nodes_df["vertex"] != "PERTURBATION"]

    # Add prior interaction sign separately where the original edge can be matched.
    pkn_sign = pkn.drop_duplicates(["source", "target"]).set_index(["source", "target"])["sign"]
    edges_df["pkn_interaction_sign"] = [
        pkn_sign.get((source, target), np.nan)
        for source, target in zip(edges_df["source"], edges_df["target"])
    ]
    edges_df.to_csv(RES / "carnival_edges.csv", index=False)
    nodes_df.to_csv(RES / "carnival_nodes.csv", index=False)

    first, second = NET_CONTRASTS
    active = {
        contrast: set(nodes_df.loc[nodes_df[contrast] != 0, "vertex"])
        for contrast in NET_CONTRASTS
    }
    node_sets = {
        "shared_active_nodes": active[first] & active[second],
        "1G_only_active_nodes": active[first] - active[second],
        "uG_only_active_nodes": active[second] - active[first],
    }
    print("Active node counts:", {key: len(value) for key, value in node_sets.items()})

    # 1. ORA of up/down DEGs.
    ora_frames = []

    for name, df in de.items():
        sig = df[df["p_val_adj"] < P_CUTOFF]
        gene_ids = df.index.astype(str)

        sets = {
            "up": sig.index[sig["avg_log2FC"] > LFC_CUTOFF],
            "down": sig.index[sig["avg_log2FC"] < -LFC_CUTOFF],
        }

    # 1b. Prioritized Sirtuin signaling and OXPHOS enrichment.
    # This block runs once, after ORA has been completed.
    priority_ora = extract_priority_pathways(ora)

    priority_ora.to_csv(
        RES / "priority_pathway_enrichment.csv",
        index=False,
    )

        # 1b. Prioritized Sirtuin signaling and OXPHOS enrichment.
    priority_ora = extract_priority_pathways(ora)

    priority_ora.to_csv(
        RES / "priority_pathway_enrichment.csv",
        index=False,
    )

    if not priority_ora.empty:
        for (contrast, direction), subset in priority_ora.groupby(
            ["contrast", "direction"],
            dropna=False,
        ):
            if pd.isna(contrast) or pd.isna(direction):
                continue

            subset.to_csv(
                RES / f"priority_{contrast}_{direction}.csv",
                index=False,
            )

        print(
            "Priority pathway enrichment results:",
            priority_ora["priority_pathway"].value_counts().to_dict(),
        )
    else:
        print(
            "No Sirtuin/OXPHOS-related terms matched the ORA results. "
            "Check the enrichment libraries, gene identifiers, and "
            "pathway-matching keywords."
        )

    # 2. Continue with your existing signed-evidence and ULM analysis here.
       # 2. Contrast-by-gene signed evidence and decoupler ULM footprints.
    # Best practice: use complete, unfiltered DE tables. Missing genes set to zero
    # here are not evidence of no effect; they may have been filtered upstream.
    stat = pd.DataFrame({k: signed_z(v) for k, v in de.items()}).T.fillna(0.0)

    if not np.isfinite(stat.to_numpy()).all():
        raise ValueError("Signed-statistic matrix contains non-finite values.")

    nets = {
        "progeny": dc.op.progeny(organism="human", top=500),
        "collectri": dc.op.collectri(organism="human"),
        "hallmark": dc.op.hallmark(organism="human"),
    }

    act_frames = []

    for kind, net in nets.items():
        scores, padj = dc.mt.ulm(stat, net, tmin=5)
        act_frames.append(tidy_activity(scores, padj, kind))

    act = (
        pd.concat(act_frames, ignore_index=True)
        if act_frames
        else pd.DataFrame()
    )

    if act.empty:
        raise RuntimeError("decoupler returned no activity scores.")

    act.to_csv(RES / "activity_ulm.csv", index=False)

    # 3. Optional per-cell PROGENy.
    try:
        progeny_on_umap(
            OUT / "pbmc_integrated_clusters.h5ad",
            nets["progeny"],
        )
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(
            "Skipping per-cell PROGENy due to input/API issue: "
            f"{exc}"
        )

    # 3b. Add the Sirtuin/OXPHOS per-cell scoring call here
    # after implementing its scoring function.

    for contrast in NET_CONTRASTS:
        plot_network(
            edges_df, nodes_df, contrast,
            RES / f"carnival_network_{contrast}.png"
        )
    # 5. Enrichment of inferred network nodes.
    # Exploratory: conditional on the pre-pruned PKN.
    pkn_bg = (
        set(sub["source"]) | set(sub["target"])
    ) - {"PERTURBATION"}

    net_ora_frames = []

    for label, nodes in node_sets.items():
        result = run_ora(
            nodes,
            pkn_bg,
            "carnival",
            label,
            libs,
        )
        if not result.empty:
            net_ora_frames.append(result)

    net_ora = (
        pd.concat(net_ora_frames, ignore_index=True)
        if net_ora_frames
        else empty_ora_frame()
    )

    net_ora.to_csv(
        RES / "ora_carnival_nodes.csv",
        index=False,
    )

    for label in node_sets:
        plot_ora(net_ora, "carnival", label)

    print(f"Done. Results in {RES}/")

if __name__ == "__main__":
    main()