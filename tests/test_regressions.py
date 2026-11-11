"""Regression tests for bugs fixed in the solver scripts (each fix has its own commit)."""
import os
import random
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- flow-network.py: Ford-Fulkerson must use residual (reverse) edges -----------------------------------------
@pytest.mark.parametrize("seed", range(100))
def test_flow_network_finds_true_maximum_flow(seed):
    import networkx as nx
    import flow_network
    random.seed(seed)
    np.random.seed(seed)
    vis = flow_network.FlowNetworkVisualizer()
    vis.generate_network(8, 0.3)   # the desktop defaults
    vis.find_max_flow()
    g = nx.DiGraph()
    for (u, v), c in vis.capacities.items():
        g.add_edge(u, v, capacity=c)
    sink = max(vis.graph.nodes())
    assert vis.max_flow_value == nx.maximum_flow_value(g, 0, sink)
    for edge, f in vis.flows.items():          # capacity and conservation constraints
        assert 0 <= f <= vis.capacities[edge]
    for n in vis.graph.nodes():
        if n not in (0, sink):
            inflow = sum(vis.flows[(u, n)] for u in vis.graph.predecessors(n))
            outflow = sum(vis.flows[(n, v)] for v in vis.graph.successors(n))
            assert inflow == outflow
    matplotlib.pyplot.close("all")
