"""Real-valued GA on twelve 2-D 'creative' and practical fitness landscapes.

Importable name for `genetic-solver.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("genetic-solver.py")

Problem = _mod.Problem
CreativeGeneticSolver = _mod.CreativeGeneticSolver
create_creative_problems = _mod.create_creative_problems

if __name__ == "__main__":
    run_as_main("genetic-solver.py")
