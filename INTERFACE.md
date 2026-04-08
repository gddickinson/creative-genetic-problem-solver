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

### Solver Scripts (underscore-named for import, hyphenated originals preserved)
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

### Tests
- **`tests/test_ga_core.py`** -- Smoke tests for shared GA operators

### Config Files
- `requirements.txt` -- pinned dependencies
- `.gitignore` -- excludes generated output and Python artifacts
- `__init__.py` -- package docstring listing all modules
