# test_network.py
import pytest
from workflows.network import NetworkBuilder

def test_graph_construction():
    builder = NetworkBuilder()
    graph = builder.create_graph(nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C")])
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2

# test_pathways.py
import pytest
from workflows.pathways import PathwayEnrichment

def test_pathway_enrichment():
    enrichment = PathwayEnrichment(database="KEGG")
    results = enrichment.analyze(gene_list=["TP53", "EGFR"])
    assert "p_value" in results.columns

# test_propagation.py
import pytest
from workflows.propagation import SignalPropagator

def test_signal_propagation():
    propagator = SignalPropagator(alpha=0.8)
    scores = propagator.run_random_walk(start_nodes=["IL6"])
    assert scores["IL6"] > 0