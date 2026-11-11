"""Reference numbers for web/static/flow_engine.js from flow-network.py (seeded fixture_rng).

Run: python scripts/make_flow_fixture.py [--write]
"""
import sys

import networkx as nx

from fixture_common import main, quiet
from fixture_rng import patched

import flow_network  # noqa: E402,F401

NAME = "flow"
mod = sys.modules["flow_network_orig"]


def run(seed, n, density):
    layout = nx.spring_layout
    with patched(mod, seed), quiet():
        mod.nx.spring_layout = lambda g: {}          # display only; it has its own random numbers
        try:
            vis = mod.FlowNetworkVisualizer()
            vis.generate_network(n, density)
            vis.find_max_flow()
        finally:
            mod.nx.spring_layout = layout
    mod.plt.close("all")
    edges = list(vis.graph.edges())
    return {"seed": seed, "n": n, "density": density, "edges": [[u, v, vis.capacities[(u, v)]] for u, v in edges],
            "paths": vis.paths, "flows": [[h[e] for e in edges] for h in vis.history], "value": vis.max_flow_value}


def fixture():
    return {"runs": [run(s, n, d) for s, n, d in [(1, 8, 0.3), (42, 8, 0.3), (5, 12, 0.6), (9, 5, 1.0), (17, 15, 0.4)]]}


if __name__ == "__main__":
    main(NAME, fixture)
