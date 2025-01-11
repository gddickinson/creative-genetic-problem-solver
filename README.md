# Creative Genetic Algorithm Visualizers

This repository contains a collection of interactive visualizations demonstrating genetic algorithms solving various complex problems. Each script provides real-time visualization of how genetic algorithms evolve solutions to challenging computational problems.

## Overview

The collection includes:

1. **Flow Network Optimizer** (`flow-network.py`)
   - Visualizes maximum flow in networks
   - Shows network bottleneck identification
   - Demonstrates flow path optimization

2. **Game Theory Evolution** (`game-evolution.py`)
   - Simulates evolution of game strategies
   - Shows population dynamics
   - Visualizes strategy performance over time

3. **Generic Problem Solver** (`genetic-solver.py`)
   - Multi-purpose genetic algorithm framework
   - Customizable fitness functions
   - Real-time optimization visualization

4. **Maze Pathfinder** (`maze-pathfinder.py`)
   - Generates and solves random mazes
   - Compares different pathfinding approaches
   - Shows evolution of path solutions

5. **Neural Network Visualizer** (`neural-vis.py`)
   - Demonstrates neural network evolution
   - Shows network architecture adaptation
   - Visualizes learning progress

6. **Packing Optimizer** (`packing-optimizer.py`)
   - Solves 2D packing problems
   - Handles circles and rectangles
   - Shows space utilization optimization

7. **Portfolio Optimizer** (`portfolio-optimizer.py`)
   - Demonstrates investment portfolio optimization
   - Shows efficient frontier calculation
   - Visualizes risk-return tradeoffs

8. **Reaction-Diffusion Simulator** (`reaction-diffusion.py`)
   - Simulates pattern formation
   - Shows emergence of complex patterns
   - Visualizes system evolution

9. **Traveling Salesperson Solver** (`travelling-sales-person-solver.py`)
   - Solves the classic TSP problem
   - Shows route optimization
   - Visualizes distance minimization

10. **Vehicle Router** (`vehicle-routing-optimizer.py`)
    - Handles multiple vehicle constraints
    - Shows time window optimization
    - Visualizes route planning

## Requirements

```bash
pip install numpy matplotlib networkx pandas seaborn
```

## Usage

Each script can be run independently. They feature interactive configuration and real-time visualization:

```bash
python script_name.py
```

For example:
```bash
python maze-pathfinder.py
```

Most visualizers allow you to configure:
- Problem size/complexity
- Population size
- Number of generations
- Algorithm parameters

## Features

Common features across all visualizers:

- **Interactive Configuration**: Set parameters before running
- **Real-time Visualization**: Watch solutions evolve
- **Multiple Metrics**: Track different aspects of evolution
- **Final Analysis**: Detailed results and suggestions
- **Educational Tools**: Great for learning about genetic algorithms

## Implementation Details

Each solver uses genetic algorithms with:
- Population-based evolution
- Fitness-based selection
- Crossover and mutation operators
- Constraint handling
- Multi-objective optimization

## Contributing

Feel free to contribute by:
- Adding new problem types
- Improving visualizations
- Enhancing algorithm efficiency
- Adding new features

## License

MIT License - feel free to use and modify for your own projects.

## Acknowledgments

These visualizers were created to demonstrate the power and versatility of genetic algorithms in solving complex optimization problems. They serve both educational and practical purposes.
