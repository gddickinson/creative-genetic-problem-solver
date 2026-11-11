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


# --- vehicle-routing-optimizer.py: insertion heuristic must restore the route it probed --------------------------
@pytest.mark.parametrize("seed", range(5))
def test_vrp_initial_solution_assigns_each_location_once(seed):
    import vehicle_routing_optimizer as vrp
    random.seed(seed)
    opt = vrp.VehicleRoutingOptimizer(12, 3)
    for loc in opt.locations[1:]:            # make every insertion feasible so only the heuristic is tested
        loc.time_window = (0, 1e9)
    for v in opt.vehicles:
        v.capacity = 1e9
    sol = opt.create_initial_solution()
    assigned = sorted(i for v in sol for i in v.route)
    assert assigned == list(range(1, 12))
    matplotlib.pyplot.close("all")


def _vrp(seed, **kw):
    import contextlib
    import io
    import vehicle_routing_optimizer as vrp
    random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = vrp.VehicleRoutingOptimizer(20, 3, **kw)
        sol = opt.create_initial_solution()
    matplotlib.pyplot.close("all")
    return opt, sol


def test_vrp_speed_default_keeps_old_behaviour_and_faster_vehicles_serve_more():
    slow, slow_sol = _vrp(3)
    same, same_sol = _vrp(3, speed=1.0)
    assert [v.route for v in slow_sol] == [v.route for v in same_sol]
    fast, fast_sol = _vrp(3, speed=20.0)
    assert sum(len(v.route) for v in fast_sol) > sum(len(v.route) for v in slow_sol)
