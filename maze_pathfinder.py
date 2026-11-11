"""Maze generation (recursive backtracking) solved by A*, BFS wave propagation and a GA.

Importable name for `maze-pathfinder.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("maze-pathfinder.py")

MazeVisualizer = _mod.MazeVisualizer

if __name__ == "__main__":
    run_as_main("maze-pathfinder.py")
