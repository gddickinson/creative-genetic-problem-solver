"""Reaction-diffusion pattern simulator."""

# Underscore-renamed version of reaction-diffusion.py for valid Python imports.

import importlib.util
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "reaction_diffusion_orig",
    os.path.join(_dir, "reaction-diffusion.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ReactionDiffusionSimulator = _mod.ReactionDiffusionSimulator
