"""Reference numbers for web/static/creative_engine.js from genetic-solver.py (seeded fixture_rng).

Run: python scripts/make_creative_fixture.py [--write]
"""
import sys

import numpy as np

from fixture_common import main, quiet
from fixture_rng import patched

import genetic_solver  # noqa: E402,F401

NAME = "creative"
mod = sys.modules["genetic_solver_orig"]
POINTS = [(0.1, 0.2), (0.5, 0.5), (0.73, 0.31), (0.97, 0.88), (0.0, 1.0), (0.333, 0.666)]


def points_in(bounds):
    return [[lo0 + (hi0 - lo0) * u, lo1 + (hi1 - lo1) * v] for (u, v) in POINTS
            for (lo0, hi0), (lo1, hi1) in [bounds]]


def run(problem_index, seed, gens, **kw):
    problem = mod.create_creative_problems()[problem_index]
    with patched(mod, seed), quiet():
        solver = mod.CreativeGeneticSolver(**kw)
        best, fit = solver.evolve(problem, generations=gens, visualize=False)
    mod.plt.close("all")
    return {"problem": problem_index, "seed": seed, "gens": gens, "params": kw, "history": solver.history,
            "diversity": solver.diversity_history, "best": best, "best_fitness": fit}


def fixture():
    problems = mod.create_creative_problems()
    values = [{"name": p.name, "points": points_in(p.bounds),
               "values": [float(p.fitness_func(np.array(x))) for x in points_in(p.bounds)]} for p in problems]
    runs = [run(2, 1, 12, population_size=40), run(4, 3, 10, population_size=30, mutation_rate=0.4, crossover_rate=0.9),
            run(7, 5, 8, population_size=20, elitism=4), run(3, 9, 6, population_size=24)]
    return {"values": values, "runs": runs}


if __name__ == "__main__":
    main(NAME, fixture)
