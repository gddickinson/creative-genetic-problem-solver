"""Catalogue of the repository's ten solvers for the web interface.

Titles and one-line summaries follow README.md; the default parameters are read from the Python signatures with
``inspect`` so the web pages show the same defaults as the desktop scripts.
"""
import inspect
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import flow_network  # noqa: E402
import game_evolution  # noqa: E402
import genetic_solver  # noqa: E402
import maze_pathfinder  # noqa: E402
import neural_vis  # noqa: E402
import packing_optimizer  # noqa: E402
import portfolio_optimizer  # noqa: E402
import reaction_diffusion  # noqa: E402
import travelling_sales_person_solver as tsp  # noqa: E402
import vehicle_routing_optimizer as vrp  # noqa: E402

# route, title, script, (callables whose keyword defaults are the desktop parameters), uses a GA, README summary
SOLVERS = [
    ("tsp", "Travelling Salesperson", "travelling-sales-person-solver.py", [tsp.TSPSolver.__init__, tsp.TSPSolver.evolve], True,
     "Solves the classic TSP problem; shows route optimization and distance minimization."),
    ("creative", "Generic Problem Solver", "genetic-solver.py", [genetic_solver.CreativeGeneticSolver.__init__, genetic_solver.CreativeGeneticSolver.evolve], True,
     "Multi-purpose genetic algorithm framework with customizable fitness functions."),
    ("vrp", "Vehicle Router", "vehicle-routing-optimizer.py", [vrp.VehicleRoutingOptimizer.__init__, vrp.VehicleRoutingOptimizer.optimize], True,
     "Handles multiple vehicle constraints and time windows; visualizes route planning."),
    ("packing", "Packing Optimizer", "packing-optimizer.py", [packing_optimizer.PackingOptimizer.__init__, packing_optimizer.PackingOptimizer.animate], True,
     "Solves 2D packing problems with circles and rectangles."),
    ("portfolio", "Portfolio Optimizer", "portfolio-optimizer.py", [portfolio_optimizer.PortfolioOptimizer.__init__, portfolio_optimizer.PortfolioOptimizer.optimize_portfolio], True,
     "Investment portfolio optimization; visualizes risk-return tradeoffs."),
    ("neural", "Neural Network Visualizer", "neural-vis.py", [neural_vis.NeuralNetVisualizer.__init__, neural_vis.NeuralNetVisualizer.genetic_train, neural_vis.NeuralNetVisualizer.generate_data], True,
     "Network weights evolved by a genetic algorithm; visualizes learning progress."),
    ("game", "Game Theory Evolution", "game-evolution.py", [game_evolution.GameEvolutionVisualizer.__init__, game_evolution.GameEvolutionVisualizer.animate], True,
     "Simulates evolution of game strategies and population dynamics."),
    ("maze", "Maze Pathfinder", "maze-pathfinder.py", [maze_pathfinder.MazeVisualizer.__init__], True,
     "Generates and solves random mazes, comparing A*, wave propagation and a genetic algorithm."),
    ("flow", "Flow Network Optimizer", "flow-network.py", [flow_network.FlowNetworkVisualizer.generate_network], False,
     "Visualizes maximum flow in networks and identifies bottlenecks (Ford-Fulkerson, not a GA)."),
    ("rd", "Reaction-Diffusion Simulator", "reaction-diffusion.py", [reaction_diffusion.ReactionDiffusionSimulator.__init__, reaction_diffusion.ReactionDiffusionSimulator.animate], False,
     "Simulates pattern formation (Gray-Scott model, not a GA)."),
]


def defaults(funcs):
    out = {}
    for f in funcs:
        for name, p in inspect.signature(f).parameters.items():
            if name != "self" and p.default is not inspect.Parameter.empty:
                out[name] = p.default
    return out


def catalogue():
    return [{"route": r, "title": t, "script": s, "is_ga": ga, "summary": d, "defaults": defaults(fs)}
            for r, t, s, fs, ga, d in SOLVERS]


def creative_problems():
    return [{"index": i, "name": p.name, "bounds": [list(b) for b in p.bounds]}
            for i, p in enumerate(genetic_solver.create_creative_problems())]


def portfolio_market(n_assets: int):
    """Market data exactly as the desktop app generates it (NumPy, np.random.seed(42) inside generate_sample_data)."""
    import contextlib
    import io

    import matplotlib.pyplot as plt
    with contextlib.redirect_stdout(io.StringIO()):
        opt = portfolio_optimizer.PortfolioOptimizer(n_assets=n_assets, population_size=10)
    plt.close("all")
    return {"n_assets": n_assets, "names": opt.asset_names, "expected_returns": opt.expected_returns.tolist(),
            "cov_matrix": opt.cov_matrix.tolist(), "risk_free_rate": opt.risk_free_rate}
