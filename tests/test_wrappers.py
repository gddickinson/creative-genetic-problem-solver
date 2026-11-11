"""The underscore modules are thin wrappers: same objects as the hyphenated originals, and runnable as scripts.

Regression for: five underscore modules were stale hand copies (neural_vis lacked the 'squares' pattern,
game_evolution lacked the final analysis), and `python packing_optimizer.py` (etc.) silently did nothing.
"""
import os
import runpy
import sys

import matplotlib
import pytest

matplotlib.use("Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from _original_loader import load  # noqa: E402

PAIRS = {
    "flow_network": ("flow-network.py", ["FlowNetworkVisualizer"]),
    "game_evolution": ("game-evolution.py", ["Strategy", "GameEvolutionVisualizer"]),
    "genetic_solver": ("genetic-solver.py", ["Problem", "CreativeGeneticSolver", "create_creative_problems"]),
    "maze_pathfinder": ("maze-pathfinder.py", ["MazeVisualizer"]),
    "neural_vis": ("neural-vis.py", ["Layer", "NeuralNetVisualizer"]),
    "packing_optimizer": ("packing-optimizer.py", ["PackingItem", "PackingOptimizer"]),
    "portfolio_optimizer": ("portfolio-optimizer.py", ["Portfolio", "PortfolioOptimizer"]),
    "reaction_diffusion": ("reaction-diffusion.py", ["ReactionDiffusionSimulator"]),
    "travelling_sales_person_solver": ("travelling-sales-person-solver.py", ["City", "TSPSolver", "create_example_problem"]),
    "vehicle_routing_optimizer": ("vehicle-routing-optimizer.py", ["Location", "Vehicle", "VehicleRoutingOptimizer"]),
}


@pytest.mark.parametrize("module", sorted(PAIRS))
def test_wrapper_reexports_original(module):
    original, names = PAIRS[module]
    wrapper = __import__(module)
    orig = load(original)
    for name in names:
        assert getattr(wrapper, name) is getattr(orig, name)


@pytest.mark.parametrize("module", sorted(PAIRS))
def test_wrapper_runs_original_as_script(module, monkeypatch):
    calls = []
    monkeypatch.setattr(runpy, "run_path", lambda path, run_name=None: calls.append((os.path.basename(path), run_name)))
    runpy.run_module(module, run_name="__main__")
    assert (PAIRS[module][0], "__main__") in calls


def test_neural_wrapper_has_squares_pattern():
    import neural_vis
    net = neural_vis.NeuralNetVisualizer([2, 3, 1])
    X, y = net.generate_data(200, "squares")
    assert X[:100].std() > 0 and y[100:].min() == 1
