"""Smoke tests for the shared GA core module."""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ga_core import (
    GAConfig,
    tournament_selection,
    roulette_selection,
    blend_crossover,
    order_crossover,
    gaussian_mutation,
    swap_mutation,
    elitist_replacement,
    calculate_diversity,
    run_ga,
)


def test_ga_config_defaults():
    """Test GAConfig has sensible defaults."""
    config = GAConfig()
    assert config.population_size == 100
    assert config.mutation_rate == 0.1
    assert config.crossover_rate == 0.7
    assert config.elitism_count == 2
    assert config.tournament_size == 5
    assert config.generations == 100


def test_tournament_selection():
    """Test tournament selection returns a valid individual."""
    population = [np.array([i]) for i in range(10)]
    fitness = np.array([float(i) for i in range(10)])
    selected = tournament_selection(population, fitness, tournament_size=3, maximize=True)
    assert isinstance(selected, np.ndarray)


def test_roulette_selection():
    """Test roulette selection returns a valid individual."""
    population = [np.array([i]) for i in range(10)]
    fitness = np.array([float(i) for i in range(10)])
    selected = roulette_selection(population, fitness)
    assert isinstance(selected, np.ndarray)


def test_blend_crossover_produces_children():
    """Test blend crossover produces two children."""
    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([4.0, 5.0, 6.0])
    c1, c2 = blend_crossover(p1, p2, crossover_rate=1.0)
    assert c1.shape == p1.shape
    assert c2.shape == p2.shape
    # Children should be between parents (blend)
    assert np.all(c1 >= np.minimum(p1, p2) - 0.01)
    assert np.all(c1 <= np.maximum(p1, p2) + 0.01)


def test_blend_crossover_no_op():
    """Test blend crossover with rate=0 returns copies."""
    p1 = np.array([1.0, 2.0])
    p2 = np.array([3.0, 4.0])
    c1, c2 = blend_crossover(p1, p2, crossover_rate=0.0)
    np.testing.assert_array_equal(c1, p1)
    np.testing.assert_array_equal(c2, p2)


def test_order_crossover():
    """Test order crossover produces a valid permutation."""
    parent1 = list(range(10))
    parent2 = list(range(9, -1, -1))
    child = order_crossover(parent1, parent2)
    assert sorted(child) == list(range(10))  # Must be a valid permutation


def test_gaussian_mutation_respects_bounds():
    """Test gaussian mutation stays within bounds."""
    individual = np.array([5.0, 5.0])
    bounds = [(0, 10), (0, 10)]
    mutated = gaussian_mutation(individual, mutation_rate=1.0, mutation_strength=100.0, bounds=bounds)
    assert mutated[0] >= 0 and mutated[0] <= 10
    assert mutated[1] >= 0 and mutated[1] <= 10


def test_swap_mutation():
    """Test swap mutation keeps all elements."""
    route = list(range(10))
    mutated = swap_mutation(route.copy(), mutation_rate=1.0)
    assert sorted(mutated) == list(range(10))


def test_elitist_replacement_maximize():
    """Test elitist replacement selects best individuals."""
    population = [f"ind_{i}" for i in range(10)]
    fitness = np.array([float(i) for i in range(10)])
    elites = elitist_replacement(population, fitness, elitism_count=2, maximize=True)
    assert len(elites) == 2
    assert "ind_9" in elites
    assert "ind_8" in elites


def test_elitist_replacement_minimize():
    """Test elitist replacement selects worst (best for minimization)."""
    population = [f"ind_{i}" for i in range(10)]
    fitness = np.array([float(i) for i in range(10)])
    elites = elitist_replacement(population, fitness, elitism_count=2, maximize=False)
    assert "ind_0" in elites
    assert "ind_1" in elites


def test_calculate_diversity():
    """Test diversity calculation returns positive for diverse population."""
    population = np.random.randn(20, 5)
    div = calculate_diversity(population)
    assert div > 0


def test_calculate_diversity_identical():
    """Test diversity is zero for identical individuals."""
    population = np.ones((10, 3))
    div = calculate_diversity(population)
    assert div == 0.0


def test_run_ga_simple_maximization():
    """Test run_ga can find maximum of a simple function."""
    config = GAConfig(population_size=30, generations=20, random_seed=42)

    def fitness_func(ind):
        return -np.sum((ind - 3.0) ** 2)  # Maximum at [3, 3]

    def init_pop():
        return [np.random.uniform(-10, 10, 2) for _ in range(config.population_size)]

    def crossover_func(p1, p2):
        return blend_crossover(p1, p2, crossover_rate=0.7)

    def mutation_func(ind):
        return gaussian_mutation(ind, mutation_rate=0.2, mutation_strength=0.5,
                                bounds=[(-10, 10), (-10, 10)])

    best, best_fitness, history = run_ga(
        fitness_func, init_pop, crossover_func, mutation_func,
        config, maximize=True
    )

    assert best is not None
    assert len(history) == 20
    # The GA should improve over time
    assert history[-1] >= history[0]


def test_run_ga_simple_minimization():
    """Test run_ga can find minimum of a simple function."""
    config = GAConfig(population_size=30, generations=20, random_seed=42)

    def fitness_func(ind):
        return np.sum(ind ** 2)  # Minimum at [0, 0]

    def init_pop():
        return [np.random.uniform(-10, 10, 2) for _ in range(config.population_size)]

    def crossover_func(p1, p2):
        return blend_crossover(p1, p2, crossover_rate=0.7)

    def mutation_func(ind):
        return gaussian_mutation(ind, mutation_rate=0.2, mutation_strength=0.5,
                                bounds=[(-10, 10), (-10, 10)])

    best, best_fitness, history = run_ga(
        fitness_func, init_pop, crossover_func, mutation_func,
        config, maximize=False
    )

    assert best is not None
    assert len(history) == 20
    # The GA should improve (decrease) over time
    assert history[-1] <= history[0]


if __name__ == "__main__":
    # Run all test functions
    test_functions = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in test_functions:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
