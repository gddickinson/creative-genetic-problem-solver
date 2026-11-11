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


def test_vrp_summary_survives_a_solution_with_no_deliveries():
    import contextlib
    import io
    opt, _ = _vrp(0)                             # desktop defaults: no location fits its time window
    with contextlib.redirect_stdout(io.StringIO()) as out:
        opt.animate(generations=2, population_size=4)
    assert "On-time Deliveries: 0/0" in out.getvalue()
    matplotlib.pyplot.close("all")


# --- maze-pathfinder.py: GA mutation must continue the path from the mutation point ------------------------------
def _seen_paths(monkeypatch, mod):
    """Record every path the GA's tournament selection sees (the whole ranked population, each generation)."""
    seen, real_sample = [], mod.random.sample

    def spy(pop, k):
        if pop and isinstance(pop[0], tuple) and isinstance(pop[0][0], list):
            seen.extend(p for p, _ in pop)
        return real_sample(pop, k)
    monkeypatch.setattr(mod.random, "sample", spy)
    return seen


@pytest.mark.parametrize("seed", [10, 12])
def test_maze_ga_paths_stay_connected(seed, monkeypatch):
    import maze_pathfinder
    mod = sys.modules["maze_pathfinder_orig"]
    random.seed(seed)
    vis = maze_pathfinder.MazeVisualizer(31, 31, population_size=20)
    vis.generate_maze()
    seen = _seen_paths(monkeypatch, mod)
    vis.genetic_pathfind()
    assert seen, "the GA should run past generation 0 on this maze"
    for path in seen:
        assert path[0] == vis.start
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
    matplotlib.pyplot.close("all")


def test_maze_astar_returns_no_path_when_goal_unreachable():
    import maze_pathfinder
    vis = maze_pathfinder.MazeVisualizer(9, 9)
    vis.maze[:] = 1
    vis.maze[vis.start] = 0
    vis.maze[vis.end] = 0
    assert vis.astar_pathfind() == []
    assert vis.wave_pathfind() == []
    matplotlib.pyplot.close("all")


# --- neural-vis.py: the 'squares' dataset should be two nested squares ------------------------------------------
@pytest.mark.parametrize("n_samples", [200, 202, 150])
def test_neural_squares_are_square_outlines(n_samples):
    import neural_vis
    np.random.seed(0)
    net = neural_vis.NeuralNetVisualizer([2, 3, 1])
    X, y = net.generate_data(n_samples, "squares")
    assert len(X) == len(y) == n_samples
    for cls in (0, 1):
        P = X[y == cls]
        lo, hi = np.percentile(P, 5, axis=0), np.percentile(P, 95, axis=0)   # robust to the 0.05 jitter
        # distance of each point to the nearest side of its class's bounding box, relative to the box size
        d = np.minimum(np.abs(P - lo), np.abs(P - hi)).min(axis=1) / (hi - lo).max()
        assert np.mean(d < 0.12) > 0.9, f"class {cls}: points do not lie on a square outline"
    matplotlib.pyplot.close("all")


# --- reaction-diffusion.py: 'random' seeds must land on any grid size ---------------------------------------------
@pytest.mark.parametrize("size", [40, 100, 200])
def test_reaction_diffusion_random_seeds_near_centre(size):
    import reaction_diffusion
    random.seed(1)
    sim = reaction_diffusion.ReactionDiffusionSimulator(size, size)
    sim.initialize_pattern("random")
    ys, xs = np.nonzero(sim.V)
    assert len(xs) > 0
    assert abs(xs.mean() - size / 2) < 8 and abs(ys.mean() - size / 2) < 8
    matplotlib.pyplot.close("all")


# --- neural-vis.py: the two 'spiral' classes should interleave, not coincide ------------------------------------
def test_neural_spiral_classes_are_separate_arms():
    import neural_vis
    np.random.seed(0)
    net = neural_vis.NeuralNetVisualizer([2, 3, 1])
    X, y = net.generate_data(200, "spiral")
    a, b = X[y == 0], X[y == 1]
    # matching points of the two arms sit on opposite sides of the origin (half a turn apart)
    assert np.mean(np.sum(a * b, axis=1) < 0) > 0.9
    matplotlib.pyplot.close("all")


# --- game-evolution.py: Tit-for-Tat and Grudger behave as the menu describes ------------------------------------
def _game():
    import game_evolution
    vis = game_evolution.GameEvolutionVisualizer(4, 1)
    vis.setup_prisoners_dilemma()
    matplotlib.pyplot.close("all")
    return game_evolution, vis


def _moves(vis, me, opponent_moves):
    """Moves of strategy `me` against a fixed sequence of opponent moves (1 = cooperate)."""
    return [int(me.decide(list(opponent_moves[:k]))) for k in range(len(opponent_moves))]


def test_tit_for_tat_cooperates_first_then_copies_last_move():
    ge, vis = _game()
    tft = vis.strategies["Tit-for-Tat"]
    opp = [0, 1, 1, 0, 0, 1, 0]
    for seed in range(5):
        random.seed(seed)
        assert _moves(vis, tft, opp) == [1] + opp[:-1]


def test_grudger_cooperates_until_first_defection_then_always_defects():
    ge, vis = _game()
    grudger = vis.strategies["Grudger"]
    for seed in range(5):
        random.seed(seed)
        assert _moves(vis, grudger, [1, 1, 1, 1]) == [1, 1, 1, 1]
        assert _moves(vis, grudger, [1, 1, 0, 1, 1, 1]) == [1, 1, 1, 0, 0, 0]


def test_random_strategy_ignores_history():
    ge, vis = _game()
    rnd = vis.strategies["Random"]
    random.seed(3)
    a = [rnd.decide([0] * k) for k in range(200)]
    random.seed(3)
    b = [rnd.decide([1] * k) for k in range(200)]
    assert a == b and 60 < sum(a) < 140
