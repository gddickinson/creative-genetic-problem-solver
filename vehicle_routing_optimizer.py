"""Capacitated vehicle routing with time windows, optimised by a GA.

Importable name for `vehicle-routing-optimizer.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("vehicle-routing-optimizer.py")

Location = _mod.Location
Vehicle = _mod.Vehicle
VehicleRoutingOptimizer = _mod.VehicleRoutingOptimizer

if __name__ == "__main__":
    run_as_main("vehicle-routing-optimizer.py")
