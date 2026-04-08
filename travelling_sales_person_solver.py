"""Travelling Salesman Problem solver using genetic algorithms."""

# Underscore-renamed version of travelling-sales-person-solver.py for valid Python imports.

import importlib.util
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "tsp_solver_orig",
    os.path.join(_dir, "travelling-sales-person-solver.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

City = _mod.City
TSPSolver = _mod.TSPSolver
create_example_problem = _mod.create_example_problem
