import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from collections import deque
import heapq
from typing import List, Tuple, Set
import matplotlib.colors as mcolors

class MazeVisualizer:
    def __init__(self, width: int = 20, height: int = 20, population_size: int = 50):
        self.width = width
        self.height = height
        self.maze = np.ones((height, width))  # 1 is wall, 0 is path
        self.start = (1, 1)
        self.end = (height-2, width-2)
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 7))
        self.current_path = []
        self.visited_cells = set()
        self.algorithm_steps = []
        self.population_size = population_size
        
    def generate_maze(self):
        """Generate maze using recursive backtracking"""
        # Reset maze
        self.maze = np.ones((self.height, self.width))
        
        def carve_path(x: int, y: int):
            self.maze[y, x] = 0
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.width and 0 <= new_y < self.height and 
                    self.maze[new_y, new_x] == 1):
                    self.maze[y + dy//2, x + dx//2] = 0
                    carve_path(new_x, new_y)
        
        # Generate main maze structure
        carve_path(1, 1)
        
        # Ensure start and end are accessible
        self.maze[self.start] = 0
        self.maze[self.end] = 0
        
        # Create a few random additional paths
        for _ in range(self.width // 2):
            x = random.randrange(1, self.width-1)
            y = random.randrange(1, self.height-1)
            if self.maze[y, x] == 1:
                self.maze[y, x] = 0
    
    def astar_pathfind(self) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm"""
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        frontier = [(0, self.start)]
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}
        self.algorithm_steps = []
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            self.algorithm_steps.append((current, set(came_from.keys())))
            
            if current == self.end:
                break
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dy, current[1] + dx)
                
                if (0 <= next_pos[0] < self.height and 
                    0 <= next_pos[1] < self.width and 
                    self.maze[next_pos] == 0):
                    
                    new_cost = cost_so_far[current] + 1
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + heuristic(self.end, next_pos)
                        heapq.heappush(frontier, (priority, next_pos))
                        came_from[next_pos] = current
        
        # Reconstruct path
        current = self.end
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        return path[::-1]
    
    def wave_pathfind(self) -> List[Tuple[int, int]]:
        """Wave propagation (BFS) pathfinding"""
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        self.algorithm_steps = []
        
        while queue:
            current, path = queue.popleft()
            self.algorithm_steps.append((current, visited.copy()))
            
            if current == self.end:
                return path
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dy, current[1] + dx)
                
                if (0 <= next_pos[0] < self.height and 
                    0 <= next_pos[1] < self.width and 
                    self.maze[next_pos] == 0 and 
                    next_pos not in visited):
                    queue.append((next_pos, path + [next_pos]))
                    visited.add(next_pos)
        
        return []
    
    def genetic_pathfind(self) -> List[Tuple[int, int]]:
        """Genetic algorithm for pathfinding"""
        def create_random_path() -> List[Tuple[int, int]]:
            path = [self.start]
            current = self.start
            visited = {current}
            max_steps = self.width * self.height
            
            while len(path) < max_steps and current != self.end:
                possible_moves = []
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    next_pos = (current[0] + dy, current[1] + dx)
                    if (0 <= next_pos[0] < self.height and 
                        0 <= next_pos[1] < self.width and 
                        self.maze[next_pos] == 0):
                        possible_moves.append(next_pos)
                
                if not possible_moves:
                    break
                
                # Prefer unvisited moves and moves closer to goal
                unvisited_moves = [m for m in possible_moves if m not in visited]
                if unvisited_moves:
                    current = min(unvisited_moves, 
                                key=lambda x: abs(x[0]-self.end[0]) + abs(x[1]-self.end[1]))
                else:
                    current = random.choice(possible_moves)
                
                path.append(current)
                visited.add(current)
            
            return path
        
        def fitness(path: List[Tuple[int, int]]) -> float:
            if not path:
                return -float('inf')
            
            # Distance to goal from end of path
            end_dist = abs(path[-1][0]-self.end[0]) + abs(path[-1][1]-self.end[1])
            
            # Rewards and penalties
            reached_end = 1000 if path[-1] == self.end else 0
            length_penalty = len(path) * 0.5
            revisit_penalty = (len(path) - len(set(path))) * 2
            
            return reached_end - end_dist - length_penalty - revisit_penalty
        
        def crossover(path1: List[Tuple[int, int]], 
                     path2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            if len(path1) < 2 or len(path2) < 2:
                return path1
            
            # Find common points
            common_points = set(path1[1:-1]) & set(path2[1:-1])
            if not common_points:
                return path1
            
            # Choose random crossing point
            cross_point = random.choice(list(common_points))
            idx1 = path1.index(cross_point)
            idx2 = path2.index(cross_point)
            
            # Create new path ensuring it starts at start point
            new_path = [self.start] + path1[1:idx1] + path2[idx2:]
            return new_path
        
        # Initialize population
        population = [create_random_path() for _ in range(self.population_size)]
        best_path = None
        best_fitness = float('-inf')
        self.algorithm_steps = []
        
        # Evolution loop
        for generation in range(100):
            # Evaluate fitness
            population_fitness = [(p, fitness(p)) for p in population]
            population_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Update best path
            if population_fitness[0][1] > best_fitness:
                best_fitness = population_fitness[0][1]
                best_path = population_fitness[0][0]
            
            # Record step for visualization
            current_best = population_fitness[0][0]
            
            # Record all cells being explored by the top 5 paths
            explored_cells = set()
            for path, _ in population_fitness[:5]:
                explored_cells.update(path)
            
            # Record each position in the current best path
            for pos in current_best:
                self.algorithm_steps.append((pos, explored_cells))
            
            # Check if we've found a valid path to the end
            if best_path and best_path[-1] == self.end:
                break
            
            # Create next generation
            next_population = []
            
            # Elitism - keep best paths
            next_population.extend([p for p, _ in population_fitness[:2]])
            
            # Create rest of population
            while len(next_population) < self.population_size:
                # Tournament selection
                tournament = random.sample(population_fitness[:20], 2)
                parent1 = max(tournament, key=lambda x: x[1])[0]
                tournament = random.sample(population_fitness[:20], 2)
                parent2 = max(tournament, key=lambda x: x[1])[0]
                
                child = crossover(parent1, parent2)
                
                # Mutation: randomly redirect path
                if random.random() < 0.1:
                    mutation_point = random.randint(1, len(child)-1)
                    new_end = create_random_path()
                    child = child[:mutation_point] + new_end
                
                next_population.append(child)
            
            population = next_population
        
        return best_path if best_path and best_path[-1] == self.end else None
    
    def update_plot(self, frame: int, algorithm_name: str):
        """Update function for animation"""
        if frame >= len(self.algorithm_steps):
            return
        
        current, visited = self.algorithm_steps[frame]
        
        # Clear previous plots
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Plot original maze
        self.axes[0].imshow(self.maze, cmap='binary')
        self.axes[0].set_title(f'Original Maze')
        
        # Plot solving progress
        solve_view = self.maze.copy()
        
        # Color visited cells
        for cell in visited:
            solve_view[cell] = 0.5
        
        # Color current cell
        if current:
            solve_view[current] = 0.8
        
        # Mark start and end
        solve_view[self.start] = 0.3
        solve_view[self.end] = 0.7
        
        # Custom colormap
        colors = ['black', 'lightblue', 'yellow', 'green', 'red']
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=5)
        
        self.axes[1].imshow(solve_view, cmap=cmap)
        self.axes[1].set_title(f'{algorithm_name} - Step {frame}')
        
        # Remove axis ticks
        for ax in self.axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
    
    def solve_and_animate(self, algorithm: str = 'astar'):
        """Solve maze and create animation"""
        self.generate_maze()
        
        if algorithm.lower() == 'astar':
            path = self.astar_pathfind()
            title = 'A* Pathfinding'
        elif algorithm.lower() == 'wave':
            path = self.wave_pathfind()
            title = 'Wave Propagation'
        else:
            path = self.genetic_pathfind()
            title = 'Genetic Algorithm Pathfinding'
        
        anim = FuncAnimation(
            self.fig, 
            lambda frame: self.update_plot(frame, title),
            frames=len(self.algorithm_steps),
            interval=100,
            repeat=False
        )
        
        plt.show()
        return path

if __name__ == "__main__":
    # Create and solve maze using all three algorithms
    print("Solving with A* algorithm...")
    visualizer = MazeVisualizer(20, 20)
    path = visualizer.solve_and_animate('astar')
    
    print("\nSolving with Wave Propagation algorithm...")
    visualizer = MazeVisualizer(20, 20)
    path = visualizer.solve_and_animate('wave')
    
    print("\nSolving with Genetic algorithm...")
    visualizer = MazeVisualizer(20, 20)
    path = visualizer.solve_and_animate('genetic')