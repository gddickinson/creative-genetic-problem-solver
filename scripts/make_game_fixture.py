"""Reference numbers for web/static/game_engine.js from game-evolution.py (seeded fixture_rng).

Run: python scripts/make_game_fixture.py [--write]
"""
import sys

from fixture_common import main, quiet
from fixture_rng import patched

import game_evolution  # noqa: E402,F401

NAME = "game"
mod = sys.modules["game_evolution_orig"]
SETUP = {"prisoners_dilemma": "setup_prisoners_dilemma", "hawk_dove": "setup_hawk_dove", "stag_hunt": "setup_stag_hunt"}


def run(seed, game, pop, gens):
    with patched(mod, seed), quiet():
        vis = mod.GameEvolutionVisualizer(pop, gens)
        getattr(vis, SETUP[game])()
        vis.evolve_population()
    mod.plt.close("all")
    return {"seed": seed, "game": game, "pop": pop, "gens": gens,
            "dist": [h[0] for h in vis.history], "perf": [h[1] for h in vis.history],
            "final_p": [s.cooperate_prob for s in vis.population], "final_names": [s.name for s in vis.population]}


def fixture():
    return {"runs": [run(1, "prisoners_dilemma", 12, 5), run(2, "hawk_dove", 10, 6), run(3, "stag_hunt", 14, 4)]}


if __name__ == "__main__":
    main(NAME, fixture)
