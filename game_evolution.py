"""Evolutionary game theory: strategy populations in repeated two-player games.

Importable name for `game-evolution.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("game-evolution.py")

Strategy = _mod.Strategy
GameEvolutionVisualizer = _mod.GameEvolutionVisualizer

if __name__ == "__main__":
    run_as_main("game-evolution.py")
