# Creative Genetic Algorithm Visualizers -- Interface Map

## Project Structure

### Shared Module
- **`ga_core.py`** -- Shared GA operators extracted from individual scripts
  - `GAConfig` -- dataclass for GA hyperparameters
  - `tournament_selection()` / `roulette_selection()` -- parent selection
  - `blend_crossover()` -- real-valued crossover
  - `order_crossover()` -- permutation crossover (TSP-style)
  - `gaussian_mutation()` -- real-valued mutation with bounds
  - `swap_mutation()` -- permutation swap mutation
  - `elitist_replacement()` -- elite preservation
  - `calculate_diversity()` -- population diversity metric
  - `run_ga()` -- generic GA loop

### Solver Scripts (hyphenated originals hold the code; underscore modules are thin importable wrappers)
- **`_original_loader.py`** -- `load(filename)` imports a hyphenated script as a module (cached in `sys.modules` as
  `<stem>_orig`); `run_as_main(filename)` runs it as a script. Every underscore module re-exports its original's
  classes through it, and `python flow_network.py` runs `flow-network.py`.

| Underscore module | Hyphenated original | Main class | Problem |
|---|---|---|---|
| `flow_network.py` | `flow-network.py` | `FlowNetworkVisualizer` | Max-flow via Ford-Fulkerson |
| `game_evolution.py` | `game-evolution.py` | `GameEvolutionVisualizer` | Evolutionary game theory |
| `genetic_solver.py` | `genetic-solver.py` | `CreativeGeneticSolver` | Multi-problem GA solver |
| `maze_pathfinder.py` | `maze-pathfinder.py` | `MazeVisualizer` | Maze generation + pathfinding |
| `neural_vis.py` | `neural-vis.py` | `NeuralNetVisualizer` | GA-trained neural networks |
| `packing_optimizer.py` | `packing-optimizer.py` | `PackingOptimizer` | 2D circle/rect packing |
| `portfolio_optimizer.py` | `portfolio-optimizer.py` | `PortfolioOptimizer` | Sharpe ratio optimization |
| `reaction_diffusion.py` | `reaction-diffusion.py` | `ReactionDiffusionSimulator` | Turing patterns |
| `travelling_sales_person_solver.py` | `travelling-sales-person-solver.py` | `TSPSolver` | TSP |
| `vehicle_routing_optimizer.py` | `vehicle-routing-optimizer.py` | `VehicleRoutingOptimizer` | CVRPTW |

### Web interface (`web/`)
- **`web/app.py`** -- FastAPI app: serves `web/static/`, API under `/api` (`/info`, `/problems`, `/creative/problems`,
  `/portfolio/market/{2..12}`); `STATIC_ENDPOINTS` lists what the static site exports.
- **`web/catalog.py`** -- the ten solvers (route, title, script, GA or not, desktop defaults read with `inspect`),
  creative problem list, portfolio market data from the real `PortfolioOptimizer.generate_sample_data`.
- **`web/static/`** -- ES-module front end, no build step:
  - `core.js`, `style.css` (shared web kit, unchanged), `style_app.css`, `index.html`, `main.js` (routes), `static_api.js`
  - `lab.js` (page layout: parameter controls, run/step/reset loop, read-out), `chart.js` (line chart, canvas mapping),
    `heat.js` (heat maps), `ga_rng.js` (seeded `Rng`: mulberry32 + the `random`/`numpy.random` calls the scripts use)
  - one engine + node test + page per solver: `tsp_engine.js`/`tsp.js`, `creative_*`, `vrp_*`, `packing_*`,
    `portfolio_*`, `neural_*`, `game_*`, `maze_*`, `flow_*`, `rd_*`; `*_engine.test.mjs` read `tests/fixtures/<name>.json`
    via `util.test.mjs`
  - `home.js`, `howga.js` (how GAs work, operator table), `about.js`
- **`run_web.py`** -- starts the app on 127.0.0.1:8154.

### Scripts (`scripts/`)
- `fixture_rng.py` -- the same generator as `ga_rng.js` in Python; `patched(mod, seed)` swaps it into a solver module.
- `fixture_common.py` -- fixture helpers (`main`, `close`, `to_json`, `no_animation`, `quiet`).
- `make_<name>_fixture.py` -- run the real solver with the shared generator and write `tests/fixtures/<name>.json`.
- `build_static.py` (static site to `site/`), `screenshots.py` (Chrome smoke test / README images in `docs/img/`).

### Tests
- **`tests/test_ga_core.py`** -- Smoke tests for shared GA operators
- **`tests/test_wrappers.py`** -- underscore modules re-export the originals and run them as scripts
- **`tests/test_regressions.py`** -- regression tests for the bugs fixed in the solver scripts
- **`tests/test_web.py`** -- API, static build, fixtures still current, node engine tests

### Config Files
- `requirements.txt` -- desktop dependencies; `requirements-web.txt` -- pinned web/test dependencies
- `.github/workflows/pages.yml` -- tests, builds and publishes the static site on push to main
- `.gitignore` -- excludes generated output and Python artifacts
- `__init__.py` -- package docstring listing all modules
