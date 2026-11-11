"""Gray-Scott reaction-diffusion pattern simulator.

Importable name for `reaction-diffusion.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("reaction-diffusion.py")

ReactionDiffusionSimulator = _mod.ReactionDiffusionSimulator

if __name__ == "__main__":
    run_as_main("reaction-diffusion.py")
