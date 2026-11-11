"""Reference numbers for web/static/maze_engine.js from maze-pathfinder.py (seeded fixture_rng).

The GA's crossover picks a cell from ``random.choice(list(common_points))`` where common_points is a Python set;
set iteration order is an implementation detail, so for the fixture the generator's choice() lists those cells in
path-1 order (exactly what the JS port does). Everything else is the script's own code.
Run: python scripts/make_maze_fixture.py [--write]
"""
import inspect
import sys

from fixture_common import main, quiet
from fixture_rng import PyRandom, patched

import maze_pathfinder  # noqa: E402,F401

NAME = "maze"
mod = sys.modules["maze_pathfinder_orig"]


class MazeRandom(PyRandom):
    def choice(self, seq):
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == "crossover":
            path1 = caller.f_locals["path1"]
            seq = sorted(seq, key=path1.index)
        return super().choice(seq)


def run(seed, w, h, pop):
    sys.setrecursionlimit(20000)
    with patched(mod, seed) as gen, quiet():
        mod.random = MazeRandom(gen)
        vis = mod.MazeVisualizer(w, h, population_size=pop)
        vis.generate_maze()
        astar = vis.astar_pathfind()
        a_steps = [list(s[0]) for s in vis.algorithm_steps]
        wave = vis.wave_pathfind()
        w_steps = [list(s[0]) for s in vis.algorithm_steps]
        ga = vis.genetic_pathfind()
        ga_steps = len(vis.algorithm_steps)
    mod.plt.close("all")
    return {"seed": seed, "w": w, "h": h, "pop": pop, "maze": vis.maze.astype(int), "astar": astar, "astar_steps": a_steps,
            "wave": wave, "wave_steps": w_steps, "ga": ga, "ga_steps": ga_steps}


def fixture():
    return {"runs": [run(1, 20, 20, 50), run(10, 31, 31, 20), run(12, 31, 31, 20), run(4, 15, 11, 10)]}


if __name__ == "__main__":
    main(NAME, fixture)
