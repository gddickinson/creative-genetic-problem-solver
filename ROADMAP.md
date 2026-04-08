# Creative Genetic Algorithm Visualizers -- Roadmap

## Current State
A collection of 10 standalone Python scripts, each implementing a genetic algorithm for a different optimization problem: flow networks, game theory, mazes, neural networks, packing, portfolios, reaction-diffusion, TSP, and vehicle routing. Each script runs independently with matplotlib visualization. No shared code between scripts, no package structure, no tests. Uses numpy, matplotlib, networkx, pandas, and seaborn.

## Short-term Improvements
- [ ] Add `requirements.txt` (numpy, matplotlib, networkx, pandas, seaborn)
- [ ] Extract shared GA logic (selection, crossover, mutation, population management) into a `ga_core.py` module
- [ ] Add type hints and docstrings to all 10 scripts
- [ ] Rename hyphenated files to use underscores for valid Python imports (e.g., `flow_network.py`)
- [ ] Add CLI arguments to each script for population size, generations, and random seed
- [ ] Add `.gitignore` for any generated output files

## Feature Enhancements
- [ ] Create a unified launcher (`main.py`) with menu to select and run any solver
- [ ] Add benchmark mode that records convergence statistics across multiple runs
- [ ] Implement configurable GA operators (tournament vs roulette selection, different crossover types)
- [ ] Add parameter sweep mode to find optimal GA hyperparameters for each problem
- [ ] Implement multi-objective optimization using NSGA-II in `portfolio-optimizer.py`
- [ ] Add export of best solutions to JSON for reproducibility
- [ ] Create animated GIF output of evolution progress for each solver

## Long-term Vision
- [ ] Build a web-based interactive platform where users can configure and watch GA evolution
- [ ] Add custom problem definition support (user-defined fitness functions)
- [ ] Implement additional metaheuristics (simulated annealing, particle swarm, ant colony) for comparison
- [ ] Create a proper Python package (`genetic_solvers`) with pluggable problem/solver architecture
- [ ] Add Jupyter notebook tutorials explaining each problem and GA approach
- [ ] Benchmark against established optimization libraries (scipy.optimize, DEAP)

## Technical Debt
- [ ] 10 scripts with no shared code -- massive duplication of GA boilerplate (likely 40-60% overlap)
- [ ] All filenames use hyphens (`flow-network.py`) -- unusable as Python modules
- [ ] No package structure (`__init__.py`, proper imports) -- just flat scripts
- [ ] Each script likely re-implements selection, crossover, and mutation independently -- consolidate
- [ ] No tests for GA correctness (e.g., does TSP solution actually improve over random?)
- [ ] Visualization code mixed with algorithm logic in every script -- separate concerns
