"""Web interface for the Creative Genetic Algorithm Visualizers.

Contents:
    app.py      FastAPI app: front end in static/, JSON API under /api (problem catalogue, creative problem list,
                portfolio market data from the real Python generator); STATIC_ENDPOINTS lists what the static site exports.
    catalog.py  The ten solvers: titles, source files, whether they are GAs, desktop parameters (read from the code).
    static/     Plain ES-module front end; one page and one JS engine (ported from the Python, node-tested) per solver.
"""
