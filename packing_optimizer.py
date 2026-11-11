"""2D circle/rectangle packing optimised by a GA.

Importable name for `packing-optimizer.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("packing-optimizer.py")

PackingItem = _mod.PackingItem
PackingOptimizer = _mod.PackingOptimizer

if __name__ == "__main__":
    run_as_main("packing-optimizer.py")
