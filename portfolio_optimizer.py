"""Portfolio weights optimised by a GA for the Sharpe ratio.

Importable name for `portfolio-optimizer.py` (the original script, which holds the code).
"""
try:
    from ._original_loader import load, run_as_main
except ImportError:
    from _original_loader import load, run_as_main

_mod = load("portfolio-optimizer.py")

Portfolio = _mod.Portfolio
PortfolioOptimizer = _mod.PortfolioOptimizer

if __name__ == "__main__":
    run_as_main("portfolio-optimizer.py")
