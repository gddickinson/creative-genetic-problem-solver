import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from typing import List, Tuple
import matplotlib.patches as patches

class City:
    def __init__(self, x: float, y: float, name: str = None):
        self.x = x
        self.y = y
        self.name = name if name else f"City_{random.randint(0, 999)}"

class TSPSolver:
    def __init__(self, cities: List[City], population_size: int = 100):
        self.cities = cities
        self.population_size = population_size
        self.population = []
        self.best_route = None
        self.best_distance = float('inf')
        self.generation = 0
        self.history = []
        
        # Initialize visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.suptitle('Traveling Salesman Problem - Genetic Algorithm Solution')
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for _ in range(self.population_size):
            route = self.cities.copy()
            random.shuffle(route)
            self.population.append(route)
    
    def calculate_distance(self, route: List[City]) -> float:
        """Calculate total distance of a route"""
        total = 0
        for i in range(len(route)):
            city1 = route[i]
            city2 = route[(i + 1) % len(route)]
            total += np.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)
        return total
    
    def crossover(self, parent1: List[City], parent2: List[City]) -> List[City]:
        """Order crossover (OX) for permutation"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Get subset of parent1
        child = [None] * size
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # Fill remaining positions with cities from parent2
        parent2_cities = [city for city in parent2 if city not in child[start:end + 1]]
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = parent2_cities[j]
                j += 1
        
        return child
    
    def mutate(self, route: List[City], mutation_rate: float = 0.01) -> List[City]:
        """Swap mutation"""
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route
    
    def select_parent(self) -> List[City]:
        """Tournament selection"""
        tournament = random.sample(self.population, 5)
        return min(tournament, key=self.calculate_distance)
    
    def update_plot(self, frame):
        """Update function for animation"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot cities and current best route
        self.ax1.set_title(f'Best Route (Generation {self.generation})')
        
        # Plot all cities
        x = [city.x for city in self.cities]
        y = [city.y for city in self.cities]
        self.ax1.scatter(x, y, c='red', s=100)
        
        # Plot current best route
        if self.best_route:
            for i in range(len(self.best_route)):
                city1 = self.best_route[i]
                city2 = self.best_route[(i + 1) % len(self.best_route)]
                self.ax1.plot([city1.x, city2.x], [city1.y, city2.y], 'b-', alpha=0.6)
        
        # Add city labels
        for city in self.cities:
            self.ax1.annotate(city.name, (city.x, city.y), 
                            xytext=(5, 5), textcoords='offset points')
        
        # Plot fitness history
        self.ax2.set_title('Fitness History')
        self.ax2.plot(self.history, 'g-')
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Best Distance (lower is better)')
        
        # Add current best distance
        if self.best_distance != float('inf'):
            self.ax2.text(0.02, 0.98, f'Current Best: {self.best_distance:.2f}',
                         transform=self.ax2.transAxes, verticalalignment='top')
        
        plt.tight_layout()
    
    def evolve(self, generations: int = 100):
        """Main evolution loop"""
        self.initialize_population()
        
        def animate(frame):
            # Evolution step
            new_population = []
            
            # Elitism - keep best route
            best_route = min(self.population, key=self.calculate_distance)
            new_population.append(best_route.copy())
            
            # Create new population
            while len(new_population) < self.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            
            # Update best route
            current_best = min(self.population, key=self.calculate_distance)
            current_best_distance = self.calculate_distance(current_best)
            
            if current_best_distance < self.best_distance:
                self.best_distance = current_best_distance
                self.best_route = current_best.copy()
            
            self.history.append(self.best_distance)
            self.generation += 1
            
            # Update plot
            self.update_plot(frame)
        
        # Create animation
        anim = FuncAnimation(self.fig, animate, frames=generations,
                           interval=100, repeat=False)
        plt.show()
        
        return self.best_route, self.best_distance

# Create example problem
def create_example_problem():
    # Create cities in interesting pattern
    cities = []
    # Create circular pattern of cities with some randomness
    n_cities = 15
    radius = 10
    center_x, center_y = 10, 10
    
    # Add some cities in a rough circle
    for i in range(n_cities - 5):
        angle = (i / (n_cities - 5)) * 2 * np.pi
        x = center_x + radius * np.cos(angle) + random.uniform(-2, 2)
        y = center_y + radius * np.sin(angle) + random.uniform(-2, 2)
        cities.append(City(x, y, f"City_{i+1}"))
    
    # Add a few cities in the middle
    for i in range(5):
        x = center_x + random.uniform(-5, 5)
        y = center_y + random.uniform(-5, 5)
        cities.append(City(x, y, f"City_{n_cities-4+i}"))
    
    return cities

if __name__ == "__main__":
    # Create and solve problem
    cities = create_example_problem()
    solver = TSPSolver(cities, population_size=100)
    best_route, best_distance = solver.evolve(generations=100)
    
    print(f"\nBest route found:")
    for city in best_route:
        print(f"  {city.name}: ({city.x:.2f}, {city.y:.2f})")
    print(f"\nTotal distance: {best_distance:.2f}")