from exerkinemap.pooling.pregnancy import (
    build_gdm_exerkine_network,
    gdm_network_tables,
)


def test_gdm_network_contains_review_exerkines_and_context_nodes() -> None:
    graph = build_gdm_exerkine_network()

    assert {
        "FGF19",
        "CHEMERIN",
        "LEPTIN",
        "ADIPONECTIN",
        "IRISIN",
        "ADIPSIN",
        "FGF21",
        "FGF23",
        "GDM",
        "MATERNAL_GLYCEMIA",
        "OFFSPRING_CARDIOMETABOLIC_RISK",
    } <= set(graph)
    assert graph.nodes["IRISIN"]["gene_symbol"] == "FNDC5"


def test_review_edges_are_not_encoded_as_causal_effects() -> None:
    graph = build_gdm_exerkine_network()

    edge = graph["LEPTIN"]["ADIPONECTIN"][0]
    assert edge["relation"] == "association"
    assert edge["directionality"] == "undirected_in_source_figure"
    assert edge["cohort_weight"] is None


def test_gdm_network_tables_include_provenance() -> None:
    nodes, edges = gdm_network_tables()

    assert any(node["node"] == "GDM" for node in nodes)
    assert any(
        edge["source"] == "FGF21"
        and edge["target"] == "MATERNAL_GLYCEMIA"
        and edge["evidence_level"] == "narrative_review"
        for edge in edges
    )