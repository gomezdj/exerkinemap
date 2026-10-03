"""Review-derived GDM exerkine network with explicit evidence annotations."""

from __future__ import annotations

from typing import Any

import networkx as nx


_EXERKINE_NODES: dict[str, dict[str, str]] = {
    "FGF19": {"gene_symbol": "FGF19", "compartment": "maternal"},
    "CHEMERIN": {"gene_symbol": "RARRES2", "compartment": "maternal"},
    "LEPTIN": {"gene_symbol": "LEP", "compartment": "maternal"},
    "ADIPONECTIN": {"gene_symbol": "ADIPOQ", "compartment": "maternal"},
    "IRISIN": {"gene_symbol": "FNDC5", "compartment": "maternal"},
    "ADIPSIN": {"gene_symbol": "CFD", "compartment": "maternal"},
    "FGF21": {"gene_symbol": "FGF21", "compartment": "maternal"},
    "FGF23": {"gene_symbol": "FGF23", "compartment": "maternal"},
}

_REVIEW_ASSOCIATIONS: tuple[tuple[str, str], ...] = (
    ("FGF19", "CHEMERIN"),
    ("FGF19", "LEPTIN"),
    ("CHEMERIN", "LEPTIN"),
    ("CHEMERIN", "FGF23"),
    ("LEPTIN", "ADIPONECTIN"),
    ("LEPTIN", "IRISIN"),
    ("ADIPONECTIN", "IRISIN"),
    ("ADIPONECTIN", "ADIPSIN"),
    ("ADIPSIN", "FGF21"),
    ("IRISIN", "FGF21"),
)


def build_gdm_exerkine_network() -> nx.MultiDiGraph:
    """Build the GDM exerkine graph represented in the supplied review figure.

    Molecular edges from the figure are associations. They are not directed
    causal claims and must receive cohort-derived weights before inference.
    """
    graph = nx.MultiDiGraph(
        name="GDM ExerkineMap",
        evidence_policy=(
            "Review-derived molecular edges are associations or hypotheses; "
            "they are not causal effects."
        ),
        source_context=(
            "Mapping the exerkines network as a promising tool to address "
            "inter-generational cardiometabolic risk in GDM: a narrative review"
        ),
    )
    graph.add_node(
        "MATERNAL_EXERCISE",
        node_type="exposure",
        display_name="Maternal exercise",
        compartment="maternal",
    )
    graph.add_node(
        "GDM",
        node_type="condition",
        display_name="Gestational diabetes mellitus",
        compartment="maternal",
    )
    graph.add_node(
        "MATERNAL_GLYCEMIA",
        node_type="outcome",
        display_name="Maternal glycemia",
        compartment="maternal",
    )
    graph.add_node(
        "OFFSPRING_CARDIOMETABOLIC_RISK",
        node_type="outcome",
        display_name="Offspring cardiometabolic risk",
        compartment="offspring",
    )
    for node, attributes in _EXERKINE_NODES.items():
        graph.add_node(
            node,
            node_type="exerkine",
            display_name=node.replace("_", " ").title(),
            evidence_level="narrative_review",
            **attributes,
        )

    for source, target in _REVIEW_ASSOCIATIONS:
        graph.add_edge(
            source,
            target,
            relation="association",
            directionality="undirected_in_source_figure",
            evidence_level="narrative_review",
            cohort_weight=None,
        )

    graph.add_edge(
        "MATERNAL_EXERCISE",
        "CHEMERIN",
        relation="exercise_association",
        directionality="hypothesis_from_review_figure",
        evidence_level="narrative_review",
        cohort_weight=None,
    )
    graph.add_edge(
        "GDM",
        "MATERNAL_GLYCEMIA",
        relation="clinical_readout",
        directionality="condition_to_readout",
        evidence_level="clinical_definition",
    )
    graph.add_edge(
        "FGF21",
        "MATERNAL_GLYCEMIA",
        relation="glycemic_association",
        directionality="hypothesis_from_review_figure",
        evidence_level="narrative_review",
        cohort_weight=None,
    )
    graph.add_edge(
        "FGF21",
        "OFFSPRING_CARDIOMETABOLIC_RISK",
        relation="intergenerational_association",
        directionality="hypothesis_from_review_figure",
        evidence_level="narrative_review",
        cohort_weight=None,
    )
    graph.add_edge(
        "FGF23",
        "GDM",
        relation="gdm_association",
        directionality="hypothesis_from_review_figure",
        evidence_level="narrative_review",
        cohort_weight=None,
    )
    return graph


def gdm_network_tables(
    graph: nx.MultiDiGraph | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return serializable node and edge tables for downstream analysis."""
    graph = graph or build_gdm_exerkine_network()
    nodes = [
        {"node": node, **attributes}
        for node, attributes in graph.nodes(data=True)
    ]
    edges = [
        {"source": source, "target": target, "edge_id": edge_id, **attributes}
        for source, target, edge_id, attributes in graph.edges(keys=True, data=True)
    ]
    return nodes, edges