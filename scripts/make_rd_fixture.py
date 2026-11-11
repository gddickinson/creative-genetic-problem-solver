"""Reference numbers for web/static/rd_engine.js from reaction-diffusion.py (seeded fixture_rng).

Run: python scripts/make_rd_fixture.py [--write]
"""
import sys

from fixture_common import main, quiet
from fixture_rng import patched

import reaction_diffusion  # noqa: E402,F401

NAME = "rd"
mod = sys.modules["reaction_diffusion_orig"]


def run(seed, size, pattern, init, frames):
    with patched(mod, seed), quiet():
        sim = mod.ReactionDiffusionSimulator(size, size)
        sim.set_parameters(pattern)
        sim.initialize_pattern(init)
        v0 = float(sim.V.sum())
        metrics = []
        for _ in range(frames):
            for _ in range(sim.steps_per_frame):
                sim.step()
            metrics.append(sim.calculate_metrics())
    mod.plt.close("all")
    return {"seed": seed, "size": size, "pattern": pattern, "init": init, "frames": frames, "v0": v0, "metrics": metrics,
            "V_row": sim.V[size // 2], "U_row": sim.U[size // 3]}


def fixture():
    return {"runs": [run(1, 40, "spots", "random", 6), run(2, 30, "stripes", "center", 4),
                     run(3, 32, "waves", "stripes", 3), run(4, 25, "chaos", "checkerboard", 3)]}


if __name__ == "__main__":
    main(NAME, fixture)
