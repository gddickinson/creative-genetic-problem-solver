"""Max-flow on a random layered network (Ford-Fulkerson with BFS augmenting paths).

Importable name for `flow-network.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("flow-network.py")

FlowNetworkVisualizer = _mod.FlowNetworkVisualizer

if __name__ == "__main__":
    run_as_main("flow-network.py")
