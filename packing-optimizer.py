import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.colors as mcolors

@dataclass
class PackingItem:
    x: float
    y: float
    width: float
    height: float = None  # If None, item is circular with width as radius
    rotation: float = 0   # Rotation in degrees (for rectangles)
    
    def overlaps(self, other: 'PackingItem') -> bool:
        """Check if this item overlaps with another, with safety margin"""
        SAFETY_MARGIN = 0.2  # Increased safety margin
        
        if self.height is None and other.height is None:
            # Both are circles
            distance = np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
            return distance < (self.width + other.width + SAFETY_MARGIN)
            
        elif self.height is not None and other.height is not None:
            # Both are rectangles - use simpler but more conservative AABB check
            # Add safety margin to dimensions
            self_left = self.x - SAFETY_MARGIN/2
            self_right = self.x + self.width + SAFETY_MARGIN/2
            self_bottom = self.y - SAFETY_MARGIN/2
            self_top = self.y + self.height + SAFETY_MARGIN/2
            
            other_left = other.x - SAFETY_MARGIN/2
            other_right = other.x + other.width + SAFETY_MARGIN/2
            other_bottom = other.y - SAFETY_MARGIN/2
            other_top = other.y + other.height + SAFETY_MARGIN/2
            
            # Check for overlap in x and y directions
            x_overlap = not (self_right < other_left or self_left > other_right)
            y_overlap = not (self_top < other_bottom or self_bottom > other_top)
            
            return x_overlap and y_overlap
            
        else:
            # Mixed shapes - handle circle-rectangle collision
            if self.height is None:
                circle, rect = self, other
            else:
                circle, rect = other, self
                
            # Find closest point on rectangle to circle center
            closest_x = max(rect.x, min(circle.x, rect.x + rect.width))
            closest_y = max(rect.y, min(circle.y, rect.y + rect.height))
            
            # Calculate distance from closest point to circle center
            distance = np.sqrt(
                (circle.x - closest_x)**2 + 
                (circle.y - closest_y)**2
            )
            
            return distance < (circle.width + SAFETY_MARGIN)

class PackingOptimizer:
    def __init__(self, container_width: float, container_height: float,
                 population_size: int = 50, mutation_rate: float = 0.1):
        self.container_width = container_width
        self.container_height = container_height
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.items = []
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.history = []
        
        # Setup visualization
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 7))
        self.current_generation = 0
        
    def generate_items(self, problem_type: str = 'circles', n_items: int = 10):
        """Generate items to pack with more conservative sizing"""
        self.items = []
        # More conservative sizing
        min_size = min(self.container_width, self.container_height) / 15  # Smaller minimum
        max_size = min(self.container_width, self.container_height) / 6   # Smaller maximum
        
        print(f"\nGenerating {n_items} {problem_type}...")
        print(f"Size range: {min_size:.2f} to {max_size:.2f}")
        
        if problem_type == 'circles':
            for i in range(n_items):
                radius = random.uniform(min_size, max_size)
                self.items.append(PackingItem(0, 0, radius))
                print(f"Circle {i+1}: radius = {radius:.2f}")
        else:
            for i in range(n_items):
                width = random.uniform(min_size, max_size)
                height = random.uniform(min_size, max_size)
                self.items.append(PackingItem(0, 0, width, height))
                print(f"Rectangle {i+1}: {width:.2f} x {height:.2f}")
        
        # Calculate total area
        total_area = sum(np.pi * item.width**2 for item in self.items) if problem_type == 'circles' else \
                    sum(item.width * item.height for item in self.items)
        container_area = self.container_width * self.container_height
        print(f"\nTotal items area: {total_area:.2f}")
        print(f"Container area: {container_area:.2f}")
        print(f"Minimum required density: {(total_area/container_area)*100:.1f}%")
    
    def create_random_solution(self) -> List[PackingItem]:
        """Create a random arrangement of items with very conservative placement"""
        solution = []
        max_attempts = 200
        
        # Sort items by size (largest first)
        sorted_items = sorted(self.items, 
                            key=lambda x: x.width * (x.height if x.height else x.width),
                            reverse=True)
        
        for item in sorted_items:
            placed = False
            best_pos = None
            min_overlap_count = float('inf')
            
            # Try grid positions
            for _ in range(max_attempts):
                if item.height is None:  # Circle
                    x = random.uniform(item.width + 0.5, 
                                     self.container_width - item.width - 0.5)
                    y = random.uniform(item.width + 0.5, 
                                     self.container_height - item.width - 0.5)
                    new_item = PackingItem(x, y, item.width)
                else:  # Rectangle
                    x = random.uniform(0.5, self.container_width - item.width - 0.5)
                    y = random.uniform(0.5, self.container_height - item.height - 0.5)
                    new_item = PackingItem(x, y, item.width, item.height, 0)  # No rotation
                
                # Count overlaps
                overlap_count = sum(1 for existing in solution 
                                  if new_item.overlaps(existing))
                
                if overlap_count == 0:
                    solution.append(new_item)
                    placed = True
                    break
                elif overlap_count < min_overlap_count:
                    min_overlap_count = overlap_count
                    best_pos = new_item
            
            if not placed and best_pos is not None:
                solution.append(best_pos)
                print(f"Warning: Item {len(solution)} placed with some overlap")
        
        return solution
        solution = []
        max_attempts = 200  # More attempts for each item
        grid_size = int(np.sqrt(len(self.items))) + 1
        
        # Create a grid of potential starting positions
        cell_width = self.container_width / grid_size
        cell_height = self.container_height / grid_size
        
        potential_positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                potential_positions.append((
                    cell_width * (i + 0.5),
                    cell_height * (j + 0.5)
                ))
        
        for item in self.items:
            placed = False
            random.shuffle(potential_positions)  # Randomize grid positions
            
            # First try grid positions
            for pos_x, pos_y in potential_positions:
                if item.height is None:  # Circle
                    # Add some random offset from grid position
                    x = pos_x + random.uniform(-cell_width/4, cell_width/4)
                    y = pos_y + random.uniform(-cell_height/4, cell_height/4)
                    # Ensure within bounds
                    x = np.clip(x, item.width*1.1, self.container_width - item.width*1.1)
                    y = np.clip(y, item.width*1.1, self.container_height - item.width*1.1)
                    new_item = PackingItem(x, y, item.width)
                else:  # Rectangle
                    x = np.clip(pos_x, item.width/2, self.container_width - item.width/2)
                    y = np.clip(pos_y, item.height/2, self.container_height - item.height/2)
                    rotation = random.choice([0, 90]) if random.random() < 0.3 else 0
                    new_item = PackingItem(x, y, item.width, item.height, rotation)
                
                # Check if this position works
                valid = True
                for existing_item in solution:
                    if new_item.overlaps(existing_item):
                        valid = False
                        break
                
                if valid:
                    solution.append(new_item)
                    placed = True
                    break
            
            # If grid positions didn't work, try random positions
            if not placed:
                for _ in range(max_attempts):
                    if item.height is None:  # Circle
                        x = random.uniform(item.width*1.1, self.container_width - item.width*1.1)
                        y = random.uniform(item.width*1.1, self.container_height - item.width*1.1)
                        new_item = PackingItem(x, y, item.width)
                    else:  # Rectangle
                        x = random.uniform(0, self.container_width - item.width)
                        y = random.uniform(0, self.container_height - item.height)
                        rotation = random.choice([0, 90]) if random.random() < 0.3 else 0
                        new_item = PackingItem(x, y, item.width, item.height, rotation)
                    
                    valid = True
                    for existing_item in solution:
                        if new_item.overlaps(existing_item):
                            valid = False
                            break
                    
                    if valid:
                        solution.append(new_item)
                        placed = True
                        break
            
            if not placed:
                print(f"Warning: Could not place item {len(solution)+1} after {max_attempts} attempts")
                # Place it anyway and let evolution handle it
                if item.height is None:
                    x = random.uniform(item.width, self.container_width - item.width)
                    y = random.uniform(item.width, self.container_height - item.width)
                    solution.append(PackingItem(x, y, item.width))
                else:
                    x = random.uniform(0, self.container_width - item.width)
                    y = random.uniform(0, self.container_height - item.height)
                    solution.append(PackingItem(x, y, item.width, item.height, 0))
        
        return solution
    
    def calculate_fitness(self, solution: List[PackingItem]) -> Tuple[float, bool]:
        """Calculate fitness of a solution, returns (fitness, is_valid)"""
        # Check for containment and overlaps
        for item in solution:
            if item.height is None:  # Circle
                if (item.x - item.width < 0 or 
                    item.x + item.width > self.container_width or
                    item.y - item.width < 0 or 
                    item.y + item.width > self.container_height):
                    return float('-inf'), False
            else:  # Rectangle
                if (item.x < 0 or item.x + item.width > self.container_width or
                    item.y < 0 or item.y + item.height > self.container_height):
                    return float('-inf'), False
        
        # Check for overlaps
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                if solution[i].overlaps(solution[j]):
                    return float('-inf'), False
        
        # Calculate compactness (negative distance from center)
        center_x, center_y = self.container_width/2, self.container_height/2
        total_distance = 0
        for item in solution:
            dx = item.x - center_x
            dy = item.y - center_y
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        return -total_distance, True  # Negative because we want to minimize distance
    
    def crossover(self, parent1: List[PackingItem], 
                 parent2: List[PackingItem]) -> List[PackingItem]:
        """Create child solution from two parents"""
        child = []
        for i in range(len(self.items)):
            if random.random() < 0.5:
                child.append(PackingItem(parent1[i].x, parent1[i].y,
                                      parent1[i].width, parent1[i].height,
                                      parent1[i].rotation))
            else:
                child.append(PackingItem(parent2[i].x, parent2[i].y,
                                      parent2[i].width, parent2[i].height,
                                      parent2[i].rotation))
        return child
    
    def mutate(self, solution: List[PackingItem]) -> List[PackingItem]:
        """Randomly modify a solution with smarter mutations"""
        mutated = []
        
        # Calculate average distance between items for scaling mutations
        avg_distance = self.container_width / (len(solution) ** 0.5)
        mutation_scale = avg_distance * 0.2  # Scale mutations relative to average spacing
        
        for item in solution:
            if random.random() < self.mutation_rate:
                # Try several mutation attempts
                best_mutated = None
                best_fitness = float('-inf')
                
                for _ in range(5):  # Try 5 different mutations
                    if item.height is None:  # Circle
                        x = item.x + random.gauss(0, mutation_scale)
                        y = item.y + random.gauss(0, mutation_scale)
                        x = np.clip(x, item.width, self.container_width - item.width)
                        y = np.clip(y, item.width, self.container_height - item.width)
                        new_item = PackingItem(x, y, item.width)
                    else:  # Rectangle
                        x = item.x + random.gauss(0, mutation_scale)
                        y = item.y + random.gauss(0, mutation_scale)
                        x = np.clip(x, 0, self.container_width - item.width)
                        y = np.clip(y, 0, self.container_height - item.height)
                        rotation = item.rotation
                        if random.random() < 0.1:  # 10% chance to rotate
                            rotation = (rotation + 90) % 360
                        new_item = PackingItem(x, y, item.width, item.height, rotation)
                    
                    # Check if this mutation is better
                    temp_solution = mutated + [new_item] + solution[len(mutated)+1:]
                    fitness, _ = self.calculate_fitness(temp_solution)
                    if fitness > best_fitness:
                        best_mutated = new_item
                        best_fitness = fitness
                
                mutated.append(best_mutated if best_mutated else item)
            else:
                mutated.append(item)
        
        return mutated
    
    def evolve(self, generations: int = 100):
        """Run genetic algorithm"""
        # Initialize population
        population = [self.create_random_solution() 
                     for _ in range(self.population_size)]
        
        for generation in range(generations):
            self.current_generation = generation
            
            # Evaluate fitness
            fitness_scores = [(solution, self.calculate_fitness(solution)[0])
                            for solution in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Update best solution
            if fitness_scores[0][1] > self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_solution = fitness_scores[0][0]
            
            # Record history
            self.history.append((self.best_solution, self.best_fitness))
            
            # Create next generation
            next_population = []
            
            # Elitism
            next_population.extend([solution for solution, _ 
                                  in fitness_scores[:2]])
            
            # Create offspring
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(fitness_scores[:10], 2)
                child = self.crossover(parent1[0], parent2[0])
                child = self.mutate(child)
                next_population.append(child)
            
            population = next_population
            
            if generation % 10 == 0:
                print(f"Generation {generation}, Best Fitness: {self.best_fitness:.2f}")
    
    def update_plot(self, frame: int):
        """Update visualization with overlap detection"""
        if frame >= len(self.history):
            return
        
        solution, fitness = self.history[frame]
        
        # Skip invalid solutions
        if solution is None:
            return
        
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Plot current packing
        ax_pack = self.axes[0]
        ax_pack.set_xlim(-1, self.container_width + 1)
        ax_pack.set_ylim(-1, self.container_height + 1)
        
        # Draw container
        container = Rectangle((0, 0), self.container_width, self.container_height,
                            fill=False, color='black')
        ax_pack.add_patch(container)
        
        # Draw items and check for overlaps
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.items)))
        for i, item in enumerate(solution):
            color = colors[i]
            if item.height is None:  # Circle
                circle = Circle((item.x, item.y), item.width,
                              fill=True, alpha=0.5, color=color)
                ax_pack.add_patch(circle)
                # Draw safety margin
                circle_margin = Circle((item.x, item.y), item.width + 0.5,
                                    fill=False, linestyle='--', color=color)
                ax_pack.add_patch(circle_margin)
            else:  # Rectangle
                rect = Rectangle((item.x, item.y), item.width, item.height,
                               fill=True, alpha=0.5, color=color)
                ax_pack.add_patch(rect)
                # Draw safety margin
                margin = 0.5
                rect_margin = Rectangle((item.x - margin/2, item.y - margin/2),
                                     item.width + margin, item.height + margin,
                                     fill=False, linestyle='--', color=color)
                ax_pack.add_patch(rect_margin)
            
            # Check for overlaps and mark them
            for j, other_item in enumerate(solution):
                if i != j and item.overlaps(other_item):
                    if item.height is None:
                        ax_pack.plot(item.x, item.y, 'rx', markersize=10)
                    else:
                        ax_pack.plot(item.x + item.width/2, item.y + item.height/2, 
                                   'rx', markersize=10)
        
        ax_pack.set_title(f'Generation {frame}: Packing Layout')
        
        # Plot fitness history
        ax_fit = self.axes[1]
        fitness_history = [f for _, f in self.history[:frame+1]]
        ax_fit.plot(range(len(fitness_history)), fitness_history, 'b-')
        ax_fit.set_title('Fitness History')
        ax_fit.set_xlabel('Generation')
        ax_fit.set_ylabel('Fitness (higher is better)')
        
        plt.tight_layout()
    
    def animate(self, problem_type: str = 'circles', n_items: int = 10,
               generations: int = 100):
        """Run optimization and create animation"""
        self.generate_items(problem_type, n_items)
        print(f"\nOptimizing packing for {n_items} {problem_type}...")
        self.evolve(generations)
        
        if self.best_solution is None or self.best_fitness == float('-inf'):
            print("\nNo valid solution found!")
            print("Try:")
            print("- Reducing the number of items")
            print("- Increasing the container size")
            print("- Running for more generations")
            return
        
        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=len(self.history),
            interval=100,
            repeat=False
        )
        
        plt.show()
        
        # Print final summary
        print("\nOptimization Complete!")
        print("=====================")
        print(f"\nProblem Type: {problem_type.capitalize()} Packing")
        print(f"Number of Items: {n_items}")
        print(f"Container Size: {self.container_width} x {self.container_height}")
        print(f"\nFinal Fitness: {self.best_fitness:.2f}")
        
        # Calculate packing density
        total_area = self.container_width * self.container_height
        items_area = 0
        for item in self.best_solution:
            if item.height is None:  # Circle
                items_area += np.pi * item.width * item.width
            else:  # Rectangle
                items_area += item.width * item.height
        
        density = (items_area / total_area) * 100
        print(f"Packing Density: {density:.1f}%")
        
        if density > 70:
            print("\nExcellent packing efficiency achieved!")
        elif density > 50:
            print("\nGood packing efficiency achieved.")
        else:
            print("\nModerate packing efficiency. Consider:")
            print("- Running for more generations")
            print("- Increasing population size")
            print("- Adjusting mutation rate")

if __name__ == "__main__":
    # Problem configuration
    print("\nPacking Problem Configuration")
    print("============================")
    
    print("\nSelect problem type:")
    print("1: Circle Packing")
    print("2: Rectangle Packing")
    problem_choice = input("Enter choice (default: 1): ").strip()
    problem_type = 'rectangles' if problem_choice == '2' else 'circles'
    
    n_items = input("\nNumber of items to pack (default: 10): ").strip()
    n_items = int(n_items) if n_items else 10
    
    container_size = input("\nContainer size (width height, default: 10 10): ").strip()
    if container_size:
        width, height = map(float, container_size.split())
    else:
        width, height = 10.0, 10.0
    
    generations = input("\nNumber of generations (default: 100): ").strip()
    generations = int(generations) if generations else 100
    
    # Create and run optimizer
    optimizer = PackingOptimizer(width, height)
    optimizer.animate(problem_type, n_items, generations)