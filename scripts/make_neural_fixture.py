"""Reference numbers for web/static/neural_engine.js from neural-vis.py (seeded fixture_rng).

Run: python scripts/make_neural_fixture.py [--write]
"""
import sys

from fixture_common import main, quiet
from fixture_rng import patched

import neural_vis  # noqa: E402,F401

NAME = "neural"
mod = sys.modules["neural_vis_orig"]
PATTERNS = ["spiral", "circles", "moons", "xor", "squares", "gaussian"]


def net_json(net):
    return [{"W": L.weights, "b": L.biases[0]} for L in net]


def run(seed, pattern, sizes, n, pop, gens):
    with patched(mod, seed), quiet():
        vis = mod.NeuralNetVisualizer(sizes, population_size=pop)
        X, y = vis.generate_data(n, pattern)
        vis.genetic_train(X, y, generations=gens)
        probe = vis.forward(X[:5], vis.best_network).ravel()
    mod.plt.close("all")
    return {"seed": seed, "pattern": pattern, "sizes": sizes, "n": n, "pop": pop, "gens": gens,
            "init": net_json(vis.network), "X": X, "y": y, "history": vis.loss_history,
            "best": net_json(vis.best_network), "probe": probe}


def fixture():
    runs = [run(i + 1, p, [2, 5, 5, 1], 40, 12, 4) for i, p in enumerate(PATTERNS)]
    runs.append(run(11, "moons", [2, 8, 1], 200, 20, 6))
    return {"runs": runs}


if __name__ == "__main__":
    main(NAME, fixture)
