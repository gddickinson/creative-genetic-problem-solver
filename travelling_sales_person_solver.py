"""Travelling salesperson problem solved by a GA (order crossover, swap mutation).

Importable name for `travelling-sales-person-solver.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("travelling-sales-person-solver.py")

City = _mod.City
TSPSolver = _mod.TSPSolver
create_example_problem = _mod.create_example_problem

if __name__ == "__main__":
    run_as_main("travelling-sales-person-solver.py")
