"""Vehicle routing optimizer using genetic algorithms."""

# Underscore-renamed version of vehicle-routing-optimizer.py for valid Python imports.

import importlib.util
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "vrp_optimizer_orig",
    os.path.join(_dir, "vehicle-routing-optimizer.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Location = _mod.Location
Vehicle = _mod.Vehicle
VehicleRoutingOptimizer = _mod.VehicleRoutingOptimizer
