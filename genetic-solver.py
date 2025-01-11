import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from typing import List, Tuple, Callable
import random
from dataclasses import dataclass
import seaborn as sns

@dataclass
class Problem:
    name: str
    dimension: int
    bounds: List[Tuple[float, float]]
    fitness_func: Callable
    optimal_value: float = None

class CreativeGeneticSolver:
    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism: int = 2
    ):
        self.init_params = {
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'elitism': elitism
        }
        self.reset()
        self.reset()
        
    def reset(self):
        """Reset the solver state"""
        self.population_size = self.init_params['population_size']
        self.mutation_rate = self.init_params['mutation_rate']
        self.crossover_rate = self.init_params['crossover_rate']
        self.elitism = self.init_params['elitism']
        self.generation = 0
        self.history = []
        self.diversity_history = []
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    def initialize_population(self, problem: Problem) -> np.ndarray:
        """Initialize random population within problem bounds"""
        population = np.zeros((self.population_size, problem.dimension))
        for i, (lower, upper) in enumerate(problem.bounds):
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        return population

    def mutate(self, individual: np.ndarray, problem: Problem) -> np.ndarray:
        """Adaptive mutation based on individual's fitness"""
        mutation_strength = np.random.normal(0, 0.1, size=individual.shape)
        mask = np.random.random(individual.shape) < self.mutation_rate
        mutated = individual + mutation_strength * mask
        
        # Ensure bounds
        for i, (lower, upper) in enumerate(problem.bounds):
            mutated[i] = np.clip(mutated[i], lower, upper)
        return mutated

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Implement adaptive blend crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1, parent2
            
        alpha = np.random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def select_parents(self, population: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tournament selection with dynamic pressure"""
        tournament_size = max(2, int(0.1 * self.population_size))
        idx1 = np.random.choice(len(population), tournament_size, replace=False)
        idx2 = np.random.choice(len(population), tournament_size, replace=False)
        
        winner1 = population[idx1[np.argmax(fitness[idx1])]]
        winner2 = population[idx2[np.argmax(fitness[idx2])]]
        return winner1, winner2

    def calculate_diversity(self, population: np.ndarray) -> float:
        """Calculate population diversity using average pairwise distance"""
        distances = []
        for i in range(min(10, len(population))):  # Sample for efficiency
            for j in range(i + 1, min(10, len(population))):
                distances.append(np.linalg.norm(population[i] - population[j]))
        return np.mean(distances) if distances else 0

    def visualize_fitness_landscape(self, problem: Problem, best_solution=None):
        """Create a contour plot of the fitness landscape"""
        x = np.linspace(problem.bounds[0][0], problem.bounds[0][1], 100)
        y = np.linspace(problem.bounds[1][0], problem.bounds[1][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = problem.fitness_func(np.array([X[i,j], Y[i,j]]))
        
        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, Z, levels=20)
        plt.colorbar(label='Fitness')
        plt.title(f'Fitness Landscape: {problem.name}')
        plt.xlabel('x')
        plt.ylabel('y')
        
        if best_solution is not None:
            plt.plot(best_solution[0], best_solution[1], 'r*', markersize=15, label='Best Solution')
            plt.legend()
        
        plt.show()

    def evolve(self, problem: Problem, generations: int = 100, visualize: bool = True):
        """Main evolution loop with visualization"""
        population = self.initialize_population(problem)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        def update(frame):
            nonlocal population
            # Evaluate fitness
            fitness = np.array([problem.fitness_func(ind) for ind in population])
            
            # Update best solution
            max_idx = np.argmax(fitness)
            if fitness[max_idx] > self.best_fitness:
                self.best_fitness = fitness[max_idx]
                self.best_solution = population[max_idx].copy()
            
            # Create new population
            new_population = []
            
            # Elitism
            elite_idx = np.argsort(fitness)[-self.elitism:]
            new_population.extend(population[elite_idx])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, problem)
                child2 = self.mutate(child2, problem)
                new_population.extend([child1, child2])
            
            population = np.array(new_population[:self.population_size])
            
            # Record history
            diversity = self.calculate_diversity(population)
            self.history.append(self.best_fitness)
            self.diversity_history.append(diversity)
            
            # Update plots
            ax1.clear()
            ax2.clear()
            
            # Fitness history plot
            ax1.plot(self.history, 'b-', label='Best Fitness')
            ax1.set_title(f'Generation {frame}: Best Fitness = {self.best_fitness:.4f}')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.legend()
            
            # Population diversity plot
            ax2.plot(self.diversity_history, 'g-', label='Population Diversity')
            ax2.set_title('Population Diversity Over Time')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Diversity')
            ax2.legend()
            
            plt.tight_layout()
            
        if visualize:
            anim = FuncAnimation(fig, update, frames=generations, repeat=False, interval=100)
            plt.show()
        else:
            for _ in range(generations):
                update(_)
        
        return self.best_solution, self.best_fitness

# Example creative problems
def create_creative_problems():
    problems = []
    
    # Practical real-world problems
    
    # 1. Solar Panel Placement
    def solar_panel_efficiency(x):
        angle = x[0]  # Panel tilt angle (0-90 degrees)
        height = x[1]  # Panel height from ground (0-10 meters)
        
        # Simulate sunlight throughout the day
        total_energy = 0
        for hour in range(24):
            sun_angle = np.sin(hour / 24 * np.pi)  # Simplified sun position
            
            # Calculate energy based on panel angle and shadows
            direct_sunlight = np.cos(np.deg2rad(angle) - sun_angle)
            shadow_effect = 1 - np.exp(-height/2)  # Higher panels get less shadowing
            
            # Combine factors
            hourly_energy = max(0, direct_sunlight * shadow_effect)
            total_energy += hourly_energy
            
        return total_energy
    
    problems.append(Problem(
        name="Solar Panel Optimization",
        dimension=2,
        bounds=[(0, 90), (0, 10)],  # angle (degrees), height (meters)
        fitness_func=solar_panel_efficiency
    ))
    
    # 2. Delivery Route Optimization
    def delivery_efficiency(x):
        warehouse_dist = x[0]  # Distance from city center (0-10 km)
        route_coverage = x[1]  # Route coverage radius (0-5 km)
        
        # Calculate costs and benefits
        transport_cost = warehouse_dist * 10  # Cost increases with distance from center
        coverage_area = np.pi * route_coverage**2  # Area covered by delivery
        population_served = coverage_area * np.exp(-warehouse_dist/5)  # Population density decreases from center
        maintenance_cost = route_coverage**2 * 5  # Larger coverage = more maintenance
        
        return population_served - transport_cost - maintenance_cost
    
    problems.append(Problem(
        name="Delivery Route Planning",
        dimension=2,
        bounds=[(0, 10), (0, 5)],  # warehouse distance (km), coverage radius (km)
        fitness_func=delivery_efficiency
    ))
    
    # 3. Garden Design Optimizer
    def garden_yield(x):
        plant_density = x[0]  # Plants per square meter (0-10)
        water_amount = x[1]  # Liters per day per square meter (0-5)
        
        # Calculate factors affecting yield
        overcrowding = np.exp(-plant_density/3)  # Too many plants reduce yield
        water_efficiency = np.exp(-(water_amount-2.5)**2/2)  # Optimal water around 2.5L
        resource_usage = plant_density * water_amount
        
        total_yield = plant_density * overcrowding * water_efficiency
        cost = resource_usage * 0.1
        
        return total_yield - cost
    
    problems.append(Problem(
        name="Garden Design",
        dimension=2,
        bounds=[(0, 10), (0, 5)],  # plant density, water amount
        fitness_func=garden_yield
    ))
    
    # 4. Wind Turbine Placement
    def wind_power(x):
        height = x[0]  # Turbine height (20-100 meters)
        spacing = x[1]  # Distance between turbines (10-50 meters)
        
        # Wind speed increases with height (log law)
        wind_speed = np.log(height/2) * 2
        
        # Wake effects decrease with spacing
        wake_loss = np.exp(-spacing/30)
        
        # Construction and maintenance costs
        cost = height * 0.1 + spacing * 0.05
        
        # Power output (simplified)
        power = wind_speed**3 * (1 - wake_loss)
        
        return power - cost
    
    problems.append(Problem(
        name="Wind Turbine Layout",
        dimension=2,
        bounds=[(20, 100), (10, 50)],  # height (m), spacing (m)
        fitness_func=wind_power
    ))
    
    # 1. Multi-modal artistic pattern
    def artistic_pattern(x):
        return np.sin(5 * x[0]) * np.cos(5 * x[1]) * np.exp(-(x[0]**2 + x[1]**2)/8)
    
    problems.append(Problem(
        name="Artistic Pattern",
        dimension=2,
        bounds=[(-4, 4), (-4, 4)],
        fitness_func=artistic_pattern
    ))
    
    # 2. Creative path finding
    def path_fitness(x):
        # Create interesting landscape with multiple peaks
        return (np.sin(3 * x[0]) + np.sin(3 * x[1]) + 
                np.sin(5 * x[0] + 2 * x[1]) + 
                np.exp(-(x[0]**2 + x[1]**2)/10))
    
    problems.append(Problem(
        name="Creative Path",
        dimension=2,
        bounds=[(-5, 5), (-5, 5)],
        fitness_func=path_fitness
    ))
    
    # 3. Mandelbrot-inspired pattern
    def mandelbrot_pattern(x):
        c = complex(x[0], x[1])
        z = 0
        for i in range(20):
            z = z*z + c
            if abs(z) > 2:
                return i/20.0
        return 1.0
    
    problems.append(Problem(
        name="Mandelbrot Pattern",
        dimension=2,
        bounds=[(-2, 0.5), (-1.25, 1.25)],
        fitness_func=mandelbrot_pattern
    ))
    
    # 4. Spiral Galaxy Formation
    def galaxy_pattern(x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        theta = np.arctan2(x[1], x[0])
        spiral = np.sin(3 * theta + r)
        density = np.exp(-r/2)
        return (spiral + 1) * density
    
    problems.append(Problem(
        name="Spiral Galaxy",
        dimension=2,
        bounds=[(-4, 4), (-4, 4)],
        fitness_func=galaxy_pattern
    ))
    
    # 5. Sound Wave Synthesis
    def sound_wave(x):
        # Simulates finding optimal parameters for a complex sound wave
        fundamental = np.sin(2 * np.pi * x[0])
        harmonics = np.sin(4 * np.pi * x[1]) * 0.5
        envelope = np.exp(-abs(x[0]))
        return (fundamental + harmonics) * envelope
    
    problems.append(Problem(
        name="Sound Wave Synthesis",
        dimension=2,
        bounds=[(-2, 2), (-2, 2)],
        fitness_func=sound_wave
    ))
    
    # 6. Neural Network Architecture
    def network_topology(x):
        # Simulates finding optimal network layer sizes
        layer1_size = abs(np.sin(x[0] * np.pi)) * 100
        layer2_size = abs(np.sin(x[1] * np.pi)) * 50
        complexity_penalty = np.exp(-(layer1_size + layer2_size)/100)
        performance = np.sin(layer1_size/10) * np.sin(layer2_size/5)
        return performance * complexity_penalty
    
    problems.append(Problem(
        name="Neural Architecture",
        dimension=2,
        bounds=[(0, 2), (0, 2)],
        fitness_func=network_topology
    ))
    
    # 7. Color Harmony
    def color_harmony(x):
        # Simulates finding harmonious color combinations in HSV space
        hue1, hue2 = x[0] % 1, x[1] % 1
        harmony = np.cos(2 * np.pi * (hue1 - hue2))
        contrast = np.sin(np.pi * abs(hue1 - hue2))
        return harmony * 0.6 + contrast * 0.4
    
    problems.append(Problem(
        name="Color Harmony",
        dimension=2,
        bounds=[(0, 1), (0, 1)],
        fitness_func=color_harmony
    ))
    
    # 8. Particle System
    def particle_system(x):
        # Simulates finding optimal parameters for a particle system
        emission_rate = np.sin(x[0] * np.pi)
        lifetime = np.cos(x[1] * np.pi)
        turbulence = np.sin(x[0] * x[1] * np.pi)
        stability = np.exp(-(x[0]**2 + x[1]**2)/4)
        return emission_rate * lifetime * (turbulence + stability)
    
    problems.append(Problem(
        name="Particle System",
        dimension=2,
        bounds=[(-2, 2), (-2, 2)],
        fitness_func=particle_system
    ))
    
    return problems

if __name__ == "__main__":
    problems = create_creative_problems()
    
    # Solve each problem with a fresh solver
    for problem in problems:
        print(f"\nSolving {problem.name}...")
        solver = CreativeGeneticSolver(
            population_size=100,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism=2
        )
        best_solution, best_fitness = solver.evolve(
            problem,
            generations=50,
            visualize=True
        )
        print(f"Best solution: {best_solution}")
        print(f"Best fitness: {best_fitness}")
        
        # Visualize the fitness landscape with the best solution
        solver.visualize_fitness_landscape(problem, best_solution)