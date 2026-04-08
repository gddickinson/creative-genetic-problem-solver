"""Portfolio optimization using genetic algorithms."""

# Underscore-renamed version of portfolio-optimizer.py for valid Python imports.

import importlib.util
import os

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "portfolio_optimizer_orig",
    os.path.join(_dir, "portfolio-optimizer.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Portfolio = _mod.Portfolio
PortfolioOptimizer = _mod.PortfolioOptimizer
