"""Reference numbers for web/static/packing_engine.js from packing-optimizer.py (seeded fixture_rng).

Run: python scripts/make_packing_fixture.py [--write]
"""
import sys

from fixture_common import main, quiet
from fixture_rng import patched

import packing_optimizer  # noqa: E402,F401

NAME = "packing"
mod = sys.modules["packing_optimizer_orig"]


def items(sol):
    return [[it.x, it.y, it.width, it.height, it.rotation] for it in sol]


def run(seed, kind, n, w, h, pop, gens, rate=0.1):
    with patched(mod, seed), quiet():
        opt = mod.PackingOptimizer(w, h, population_size=pop, mutation_rate=rate)
        opt.generate_items(kind, n)
        opt.evolve(gens)
    mod.plt.close("all")
    return {"seed": seed, "kind": kind, "n": n, "w": w, "h": h, "pop": pop, "gens": gens, "rate": rate,
            "items": items(opt.items), "history": [f for _, f in opt.history],
            "best": items(opt.best_solution) if opt.best_solution else None}


def overlap_cases():
    P = mod.PackingItem
    shapes = [P(5, 5, 1), P(6.5, 5, 0.4), P(3, 3, 2, 1), P(4.95, 3.5, 1, 1), P(5.5, 4.9, 1.2, 0.5), P(1, 8, 0.7)]
    return [[i, j, bool(a.overlaps(b))] for i, a in enumerate(shapes) for j, b in enumerate(shapes) if i != j], items(shapes)


def fixture():
    pairs, shapes = overlap_cases()
    return {"overlaps": pairs, "shapes": shapes,
            "runs": [run(1, "circles", 8, 10.0, 10.0, 12, 6), run(4, "rectangles", 7, 10.0, 8.0, 10, 5, 0.5),
                     run(9, "circles", 5, 12.0, 9.0, 8, 8, 0.6)]}


if __name__ == "__main__":
    main(NAME, fixture)
