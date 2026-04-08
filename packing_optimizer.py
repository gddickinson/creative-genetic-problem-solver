"""2D bin packing optimizer using genetic algorithms."""

# Underscore-renamed version of packing-optimizer.py for valid Python imports.
# See packing-optimizer.py for the original.
# Content is identical -- created to enable import as a Python module.

import importlib.util
import os

# Load from the hyphenated original to avoid code duplication
_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "packing_optimizer_orig",
    os.path.join(_dir, "packing-optimizer.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export all public names
PackingItem = _mod.PackingItem
PackingOptimizer = _mod.PackingOptimizer

if __name__ == "__main__":
    # Delegate to original
    _mod.__name__ = "__main__"
