"""
Shared genetic algorithm core logic.

Provides reusable GA operators (selection, crossover, mutation, population management)
that can be used across all solver scripts.
"""

import numpy as np
import random
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class GAConfig:
    """Configuration for genetic algorithm parameters."""
    population_size: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_count: int = 2
    tournament_size: int = 5
    generations: int = 100
    random_seed: Optional[int] = None


def tournament_selection(
    population: list,
    fitness_scores: np.ndarray,
    tournament_size: int = 5,
    maximize: bool = True,
) -> object:
    """
    Select an individual using tournament selection.

    Args:
        population: List of individuals.
        fitness_scores: Array of fitness values corresponding to population.
        tournament_size: Number of individuals in each tournament.
        maximize: If True, select the individual with highest fitness.

    Returns:
        The selected individual.
    """
    indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    if maximize:
        winner_idx = indices[np.argmax(fitness_scores[indices])]
    else:
        winner_idx = indices[np.argmin(fitness_scores[indices])]
    return population[winner_idx]


def roulette_selection(
    population: list,
    fitness_scores: np.ndarray,
) -> object:
    """
    Select an individual using roulette wheel (fitness-proportionate) selection.

    Args:
        population: List of individuals.
        fitness_scores: Array of non-negative fitness values.

    Returns:
        The selected individual.
    """
    # Shift to non-negative
    shifted = fitness_scores - np.min(fitness_scores) + 1e-10
    probs = shifted / np.sum(shifted)
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


def blend_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend crossover for real-valued individuals.

    Args:
        parent1: First parent array.
        parent2: Second parent array.
        crossover_rate: Probability of performing crossover.

    Returns:
        Two offspring arrays.
    """
    if np.random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    alpha = np.random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2


def order_crossover(
    parent1: list,
    parent2: list,
) -> list:
    """
    Order crossover (OX) for permutation-based individuals (e.g., TSP).

    Args:
        parent1: First parent (list of items).
        parent2: Second parent (list of items).

    Returns:
        A child permutation.
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    for i in range(start, end + 1):
        child[i] = parent1[i]

    remaining = [item for item in parent2 if item not in child[start:end + 1]]
    j = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[j]
            j += 1

    return child


def gaussian_mutation(
    individual: np.ndarray,
    mutation_rate: float = 0.1,
    mutation_strength: float = 0.1,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Gaussian mutation for real-valued individuals.

    Args:
        individual: The individual to mutate.
        mutation_rate: Probability of mutating each gene.
        mutation_strength: Standard deviation of the Gaussian noise.
        bounds: Optional list of (lower, upper) bounds per dimension.

    Returns:
        The mutated individual.
    """
    noise = np.random.normal(0, mutation_strength, size=individual.shape)
    mask = np.random.random(individual.shape) < mutation_rate
    mutated = individual + noise * mask

    if bounds is not None:
        for i, (lower, upper) in enumerate(bounds):
            mutated[i] = np.clip(mutated[i], lower, upper)

    return mutated


def swap_mutation(route: list, mutation_rate: float = 0.01) -> list:
    """
    Swap mutation for permutation-based individuals.

    Args:
        route: List of items (e.g., cities).
        mutation_rate: Probability of performing a swap.

    Returns:
        The (possibly mutated) route.
    """
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route


def elitist_replacement(
    population: list,
    fitness_scores: np.ndarray,
    elitism_count: int = 2,
    maximize: bool = True,
) -> list:
    """
    Get the elite individuals from the population.

    Args:
        population: Current population.
        fitness_scores: Fitness values for each individual.
        elitism_count: Number of elites to preserve.
        maximize: If True, keep highest-fitness individuals.

    Returns:
        List of elite individuals.
    """
    if maximize:
        elite_indices = np.argsort(fitness_scores)[-elitism_count:]
    else:
        elite_indices = np.argsort(fitness_scores)[:elitism_count]
    return [population[i] for i in elite_indices]


def calculate_diversity(population: np.ndarray, sample_size: int = 10) -> float:
    """
    Calculate population diversity using average pairwise distance.

    Args:
        population: Array of individuals (each row is an individual).
        sample_size: Number of individuals to sample for efficiency.

    Returns:
        Average pairwise Euclidean distance.
    """
    n = min(sample_size, len(population))
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(np.linalg.norm(population[i] - population[j]))
    return np.mean(distances) if distances else 0.0


def run_ga(
    fitness_func: Callable,
    init_population_func: Callable,
    crossover_func: Callable,
    mutation_func: Callable,
    config: GAConfig,
    maximize: bool = True,
    callback: Optional[Callable] = None,
) -> Tuple[object, float, list]:
    """
    Run a generic genetic algorithm loop.

    Args:
        fitness_func: Function that takes an individual and returns a fitness score.
        init_population_func: Function that returns a list of individuals.
        crossover_func: Function(parent1, parent2) -> child or (child1, child2).
        mutation_func: Function(individual) -> mutated individual.
        config: GA configuration parameters.
        maximize: If True, maximize fitness; otherwise minimize.
        callback: Optional function called each generation with (gen, best, best_fitness).

    Returns:
        Tuple of (best_individual, best_fitness, history).
    """
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    population = init_population_func()
    best_individual = None
    best_fitness = float('-inf') if maximize else float('inf')
    history = []

    for generation in range(config.generations):
        # Evaluate fitness
        fitness_scores = np.array([fitness_func(ind) for ind in population])

        # Update best
        if maximize:
            idx = np.argmax(fitness_scores)
            if fitness_scores[idx] > best_fitness:
                best_fitness = fitness_scores[idx]
                best_individual = population[idx]
        else:
            idx = np.argmin(fitness_scores)
            if fitness_scores[idx] < best_fitness:
                best_fitness = fitness_scores[idx]
                best_individual = population[idx]

        history.append(best_fitness)

        if callback:
            callback(generation, best_individual, best_fitness)

        # Elitism
        elites = elitist_replacement(
            population, fitness_scores, config.elitism_count, maximize
        )
        new_population = list(elites)

        # Generate offspring
        while len(new_population) < config.population_size:
            parent1 = tournament_selection(
                population, fitness_scores, config.tournament_size, maximize
            )
            parent2 = tournament_selection(
                population, fitness_scores, config.tournament_size, maximize
            )
            result = crossover_func(parent1, parent2)
            if isinstance(result, tuple):
                child1, child2 = result
                child1 = mutation_func(child1)
                child2 = mutation_func(child2)
                new_population.extend([child1, child2])
            else:
                child = mutation_func(result)
                new_population.append(child)

        population = new_population[:config.population_size]

    return best_individual, best_fitness, history
