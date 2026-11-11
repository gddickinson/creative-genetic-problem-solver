"""Reference numbers for web/static/portfolio_engine.js from portfolio-optimizer.py (seeded fixture_rng).

Run: python scripts/make_portfolio_fixture.py [--write]
"""
import sys

from fixture_common import main, quiet
from fixture_rng import patched

import portfolio_optimizer  # noqa: E402,F401

NAME = "portfolio"
mod = sys.modules["portfolio_optimizer_orig"]


def run(n_assets, pop, gens, frontier_points):
    with patched(mod, 1), quiet():
        opt = mod.PortfolioOptimizer(n_assets=n_assets, population_size=pop)
        opt.calculate_efficient_frontier(n_points=frontier_points)
        opt.optimize_portfolio(gens)
    mod.plt.close("all")
    return {"n_assets": n_assets, "pop": pop, "gens": gens, "frontier_points": frontier_points,
            "expected_returns": opt.expected_returns, "cov_matrix": opt.cov_matrix,
            "frontier": opt.efficient_frontier.tolist(),
            "sharpe": [h["best_portfolio"].sharpe_ratio for h in opt.history],
            "best_weights": opt.best_portfolio.weights, "best_return": opt.best_portfolio.expected_return,
            "best_volatility": opt.best_portfolio.volatility}


def fixture():
    return {"runs": [run(5, 40, 15, 8), run(3, 25, 10, 4), run(8, 30, 6, 3)]}


if __name__ == "__main__":
    main(NAME, fixture)
