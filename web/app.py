"""FastAPI web app for the Creative Genetic Algorithm Visualizers: serves the front end in static/ and a small JSON API.

The solvers themselves run in the browser (JS ports in static/*_engine.js, node-tested against the Python). The API
supplies what comes straight from the Python code: the solver catalogue with the desktop defaults, the creative
problem list, and portfolio market data from the real NumPy generator.

Run with `python run_web.py` (http://127.0.0.1:8154, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web import catalog

STATIC = Path(__file__).resolve().parent / "static"
PORTFOLIO_SIZES = range(2, 13)

app = FastAPI(title="Creative Genetic Algorithm Visualizers")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Counts for the home page."""
    cat = catalog.catalogue()
    return {"title": "Creative Genetic Algorithm Visualizers", "solvers": len(cat),
            "ga_solvers": sum(c["is_ga"] for c in cat), "creative_problems": len(catalog.creative_problems())}


@app.get("/api/problems")
def problems():
    """The ten solvers: route, title, source script, GA or not, README summary, desktop defaults."""
    return catalog.catalogue()


@app.get("/api/creative/problems")
def creative_problems():
    """Names and bounds of the twelve problems in genetic-solver.py create_creative_problems()."""
    return catalog.creative_problems()


@app.get("/api/portfolio/market/{n_assets}")
def portfolio_market(n_assets: int):
    """Synthetic market (expected returns, covariance) from PortfolioOptimizer.generate_sample_data."""
    if n_assets not in PORTFOLIO_SIZES:
        raise HTTPException(404, f"n_assets must be {PORTFOLIO_SIZES.start}-{PORTFOLIO_SIZES.stop - 1}")
    return catalog.portfolio_market(n_assets)


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/problems", "/creative/problems"] + [f"/portfolio/market/{n}" for n in PORTFOLIO_SIZES]
