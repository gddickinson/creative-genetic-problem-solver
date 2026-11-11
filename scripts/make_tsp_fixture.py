"""Reference numbers for web/static/tsp_engine.js from travelling-sales-person-solver.py (seeded fixture_rng).

Run: python scripts/make_tsp_fixture.py [--write]
"""
from fixture_common import main, no_animation, quiet
from fixture_rng import patched

import travelling_sales_person_solver as tsp_wrapper  # noqa: E402,F401
import sys  # noqa: E402

NAME = "tsp"
mod = sys.modules["travelling_sales_person_solver_orig"]


def run(seed, pop, gens, mutation_rate=None):
    with patched(mod, seed), quiet():
        cities = mod.create_example_problem()
        solver = mod.TSPSolver(cities, population_size=pop)
        solver.update_plot = lambda frame: None
        if mutation_rate is not None:
            real = solver.mutate
            solver.mutate = lambda route, rate=0.01: real(route, mutation_rate)
        no_animation(mod)
        best, dist = solver.evolve(generations=gens)
        idx = {id(c): i for i, c in enumerate(cities)}
        mod.plt.close("all")
        return {"seed": seed, "pop": pop, "gens": gens, "mutation_rate": mutation_rate,
                "cities": [[c.x, c.y, c.name] for c in cities], "history": solver.history,
                "best_route": [idx[id(c)] for c in best], "best_distance": dist}


def operators(seed):
    """crossover / mutate / select_parent on a fixed population, same random stream."""
    with patched(mod, seed), quiet():
        cities = mod.create_example_problem()
        solver = mod.TSPSolver(cities, population_size=12)
        solver.initialize_population()
        idx = {id(c): i for i, c in enumerate(cities)}
        ids = lambda r: [idx[id(c)] for c in r]  # noqa: E731
        out = {"population": [ids(r) for r in solver.population], "steps": []}
        for _ in range(6):
            a, b = solver.select_parent(), solver.select_parent()
            child = solver.crossover(a, b)
            out["steps"].append({"a": ids(a), "b": ids(b), "child": ids(child),
                                 "mutated": ids(solver.mutate(list(child), 1.0)),
                                 "distance": solver.calculate_distance(child)})
        mod.plt.close("all")
        return out


def fixture():
    return {"runs": [run(1, 30, 25), run(7, 50, 15, 0.3)], "operators": operators(5)}


if __name__ == "__main__":
    main(NAME, fixture)
