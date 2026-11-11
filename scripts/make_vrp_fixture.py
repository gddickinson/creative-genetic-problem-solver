"""Reference numbers for web/static/vrp_engine.js from vehicle-routing-optimizer.py (seeded fixture_rng).

Run: python scripts/make_vrp_fixture.py [--write]
"""
import sys

from fixture_common import main, quiet
from fixture_rng import patched

import vehicle_routing_optimizer  # noqa: E402,F401

NAME = "vrp"
mod = sys.modules["vehicle_routing_optimizer_orig"]


def run(seed, n_loc, n_veh, speed, pop, gens):
    with patched(mod, seed), quiet():
        opt = mod.VehicleRoutingOptimizer(n_loc, n_veh, speed=speed)
        initial = opt.create_initial_solution()
        opt.optimize(generations=gens, population_size=pop)
    mod.plt.close("all")
    return {"seed": seed, "n_loc": n_loc, "n_veh": n_veh, "speed": speed, "pop": pop, "gens": gens,
            "locations": [[l.x, l.y, l.demand, l.time_window[0], l.time_window[1], l.service_time, l.priority] for l in opt.locations],
            "capacities": [v.capacity for v in opt.vehicles], "initial": [v.route for v in initial],
            "history": [h["best_fitness"] for h in opt.history],
            "best": [v.route for v in opt.best_solution] if opt.best_solution else None, "best_fitness": opt.best_fitness}


def fixture():
    return {"runs": [run(1, 12, 3, 20.0, 20, 12), run(2, 20, 3, 1.0, 10, 4), run(5, 15, 4, 10.0, 24, 10)]}


if __name__ == "__main__":
    main(NAME, fixture)
