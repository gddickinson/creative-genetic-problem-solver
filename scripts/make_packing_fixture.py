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
    shapes = [P(5, 5, 1), P(6.5, 5, 0.4), P(3, 3, 2, 1), P(4.95, 3.5, 1, 1), P(5.5, 4.9, 1.2, 0.5), P(1, 8, 0.7),
              P(3, 3, 2, 1, 90), P(4.2, 2.5, 0.5, 3, 270), P(9.5, 9.5, 1.5, 0.4, 90)]
    pairs = [[i, j, bool(a.overlaps(b)), float(a.overlap_depth(b))]
             for i, a in enumerate(shapes) for j, b in enumerate(shapes) if i != j]
    return pairs, items(shapes)


def fixture():
    pairs, shapes = overlap_cases()
    opt = mod.PackingOptimizer(10.0, 10.0)
    mod.plt.close("all")
    shape_objs = [mod.PackingItem(*row) for row in shapes]
    return {"overlaps": pairs, "shapes": shapes, "score": opt.ranking_score(shape_objs),
            "violations": opt.violations(shape_objs),
            "runs": [run(1, "circles", 8, 10.0, 10.0, 12, 6), run(4, "rectangles", 7, 10.0, 8.0, 10, 5, 0.5),
                     run(9, "circles", 5, 12.0, 9.0, 8, 8, 0.6),
                     run(0, "circles", 10, 10.0, 10.0, 16, 12, 0.3), run(6, "rectangles", 12, 10.0, 10.0, 12, 8, 0.5)]}


if __name__ == "__main__":
    main(NAME, fixture)
