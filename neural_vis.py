"""Neural network weights evolved by a GA on 2-D classification patterns.

Importable name for `neural-vis.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("neural-vis.py")

Layer = _mod.Layer
NeuralNetVisualizer = _mod.NeuralNetVisualizer

if __name__ == "__main__":
    run_as_main("neural-vis.py")
