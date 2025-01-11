import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass

@dataclass
class Layer:
    weights: np.ndarray
    biases: np.ndarray
    activation: str = 'relu'

class NeuralNetVisualizer:
    def __init__(self, layer_sizes: List[int], population_size: int = 50):
        self.layer_sizes = layer_sizes
        self.population_size = population_size
        self.network = self.initialize_network()
        self.fig = plt.figure(figsize=(15, 5))
        
        # Create subplots
        self.ax_network = plt.subplot2grid((1, 3), (0, 0))  # Network architecture
        self.ax_decision = plt.subplot2grid((1, 3), (0, 1))  # Decision boundary
        self.ax_loss = plt.subplot2grid((1, 3), (0, 2))     # Loss history
        
        self.loss_history = []
        self.decision_history = []
        self.best_network = None
        self.best_fitness = float('-inf')
        
    def initialize_network(self) -> List[Layer]:
        """Initialize network with random weights"""
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            weights = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1
            biases = np.random.randn(1, self.layer_sizes[i+1]) * 0.1
            layers.append(Layer(weights, biases))
        return layers
    
    def activation(self, x: np.ndarray, func: str = 'relu') -> np.ndarray:
        """Apply activation function"""
        if func == 'relu':
            return np.maximum(0, x)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        return x
    
    def forward(self, X: np.ndarray, network: Optional[List[Layer]] = None) -> np.ndarray:
        """Forward pass through network"""
        if network is None:
            network = self.network
            
        current = X
        for i, layer in enumerate(network):
            current = self.activation(
                np.dot(current, layer.weights) + layer.biases,
                'sigmoid' if i == len(network)-1 else 'relu'
            )
        return current
    
    def generate_data(self, n_samples: int = 200, pattern: str = 'spiral') -> Tuple[np.ndarray, np.ndarray]:
        """Generate different pattern datasets"""
        points_per_class = n_samples // 2
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples)
        
        if pattern == 'spiral':
            # Two interleaving spirals
            for i in range(2):
                t = np.linspace(0, 4*np.pi, points_per_class)
                r = t + np.random.randn(points_per_class) * 0.2
                X[i*points_per_class:(i+1)*points_per_class] = np.column_stack([
                    r*np.cos(t), r*np.sin(t)
                ])
                y[i*points_per_class:(i+1)*points_per_class] = i
                
        elif pattern == 'circles':
            # Concentric circles
            for i in range(2):
                r = i + 1 + np.random.randn(points_per_class) * 0.1
                t = np.linspace(0, 2*np.pi, points_per_class) + np.random.randn(points_per_class) * 0.2
                X[i*points_per_class:(i+1)*points_per_class] = np.column_stack([
                    r*np.cos(t), r*np.sin(t)
                ])
                y[i*points_per_class:(i+1)*points_per_class] = i
                
        elif pattern == 'moons':
            # Two half-moons
            t = np.linspace(0, np.pi, points_per_class)
            # First moon
            X[:points_per_class] = np.column_stack([
                np.cos(t), np.sin(t)
            ]) + np.random.randn(points_per_class, 2) * 0.1
            # Second moon
            X[points_per_class:] = np.column_stack([
                1 - np.cos(t), 0.5 - np.sin(t)
            ]) + np.random.randn(points_per_class, 2) * 0.1
            y[points_per_class:] = 1
            
        elif pattern == 'xor':
            # XOR pattern
            X[:points_per_class//2] = np.random.randn(points_per_class//2, 2) * 0.2 + [-1, -1]
            X[points_per_class//2:points_per_class] = np.random.randn(points_per_class//2, 2) * 0.2 + [1, 1]
            X[points_per_class:points_per_class+points_per_class//2] = np.random.randn(points_per_class//2, 2) * 0.2 + [-1, 1]
            X[points_per_class+points_per_class//2:] = np.random.randn(points_per_class//2, 2) * 0.2 + [1, -1]
            y[:points_per_class] = 0
            y[points_per_class:] = 1
            
        elif pattern == 'squares':
            # Nested squares
            def generate_square(size, offset):
                t = np.linspace(0, 4, points_per_class//4)
                points = []
                for i in range(4):
                    if i == 0:  # Bottom edge
                        points.extend(list(zip(t*size, [0]*len(t))))
                    elif i == 1:  # Right edge
                        points.extend(list(zip([size]*len(t), t*size)))
                    elif i == 2:  # Top edge
                        points.extend(list(zip((1-t)*size, [size]*len(t))))
                    else:  # Left edge
                        points.extend(list(zip([0]*len(t), (1-t)*size)))
                return np.array(points) + offset
            
            # Inner square
            X[:points_per_class] = generate_square(1, [-0.5, -0.5]) + np.random.randn(points_per_class, 2) * 0.05
            # Outer square
            X[points_per_class:] = generate_square(2, [-1, -1]) + np.random.randn(points_per_class, 2) * 0.05
            y[points_per_class:] = 1
            
        elif pattern == 'gaussian':
            # Gaussian clusters
            centers = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
            for i in range(2):
                X[i*points_per_class:(i+1)*points_per_class] = np.random.randn(points_per_class, 2) * 0.2 + centers[i]
                y[i*points_per_class:(i+1)*points_per_class] = i
        
        # Normalize data
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        return X, y
    
    def genetic_train(self, X: np.ndarray, y: np.ndarray, generations: int = 100):
        """Train network using genetic algorithm"""
        def create_random_network() -> List[Layer]:
            return [Layer(
                np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1,
                np.random.randn(1, self.layer_sizes[i+1]) * 0.1
            ) for i in range(len(self.layer_sizes) - 1)]
        
        def fitness(network: List[Layer]) -> float:
            predictions = self.forward(X, network)
            accuracy = np.mean((predictions > 0.5).flatten() == y)
            return accuracy
        
        def crossover(net1: List[Layer], net2: List[Layer]) -> List[Layer]:
            child = []
            for layer1, layer2 in zip(net1, net2):
                # Crossover weights
                mask = np.random.random(layer1.weights.shape) < 0.5
                weights = np.where(mask, layer1.weights, layer2.weights)
                
                # Crossover biases
                mask = np.random.random(layer1.biases.shape) < 0.5
                biases = np.where(mask, layer1.biases, layer2.biases)
                
                child.append(Layer(weights, biases))
            return child
        
        def mutate(network: List[Layer], rate: float = 0.1) -> List[Layer]:
            mutated = []
            for layer in network:
                # Mutate weights
                mutation_mask = np.random.random(layer.weights.shape) < 0.1
                mutation = np.random.randn(*layer.weights.shape) * rate
                weights = layer.weights.copy()
                weights[mutation_mask] += mutation[mutation_mask]
                
                # Mutate biases
                mutation_mask = np.random.random(layer.biases.shape) < 0.1
                mutation = np.random.randn(*layer.biases.shape) * rate
                biases = layer.biases.copy()
                biases[mutation_mask] += mutation[mutation_mask]
                
                mutated.append(Layer(weights, biases))
            return mutated
        
        # Initialize population
        population = [create_random_network() for _ in range(self.population_size)]
        
        self.loss_history = []
        self.decision_history = []
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [(net, fitness(net)) for net in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Update best network
            if fitness_scores[0][1] > self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_network = fitness_scores[0][0]
            
            # Record history
            self.loss_history.append(self.best_fitness)
            self.decision_history.append(self.get_decision_boundary(self.best_network))
            
            # Create next generation
            next_population = []
            
            # Elitism
            next_population.extend([net for net, _ in fitness_scores[:2]])
            
            # Create offspring
            while len(next_population) < self.population_size:
                # Tournament selection
                parents = random.sample([net for net, _ in fitness_scores[:10]], 2)
                child = crossover(parents[0], parents[1])
                child = mutate(child)
                next_population.append(child)
            
            population = next_population
            
            if generation % 10 == 0:
                print(f"Generation {generation}, Best Fitness: {self.best_fitness:.4f}")
    
    def get_decision_boundary(self, network: Optional[List[Layer]] = None) -> np.ndarray:
        """Get decision boundary for visualization"""
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.forward(X_grid, network if network else self.network)
        Z = Z.reshape(xx.shape)
        return Z
    
    def update_plot(self, frame: int, X: np.ndarray, y: np.ndarray):
        """Update function for animation"""
        # Clear plots
        self.ax_network.clear()
        self.ax_decision.clear()
        self.ax_loss.clear()
        
        # Plot network architecture
        for i, size in enumerate(self.layer_sizes):
            x = np.ones(size) * i
            y_pos = np.linspace(-size/2, size/2, size)
            self.ax_network.scatter(x, y_pos, c='blue', s=100)
            
            if i < len(self.layer_sizes) - 1:
                for j in range(size):
                    for k in range(self.layer_sizes[i + 1]):
                        weight = self.best_network[i].weights[j, k] if self.best_network else 0
                        color = 'red' if weight < 0 else 'green'
                        alpha = min(abs(weight), 1)
                        self.ax_network.plot([i, i+1], 
                                          [y_pos[j], np.linspace(-self.layer_sizes[i+1]/2, 
                                                               self.layer_sizes[i+1]/2, 
                                                               self.layer_sizes[i+1])[k]],
                                          c=color, alpha=alpha)
        
        self.ax_network.set_title('Network Architecture')
        
        # Plot decision boundary
        if self.decision_history:
            xx, yy = np.meshgrid(np.linspace(-3, 3, 100),
                                np.linspace(-3, 3, 100))
            self.ax_decision.contourf(xx, yy, self.decision_history[frame], 
                                    levels=20, cmap='RdBu')
            self.ax_decision.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', 
                                   edgecolors='black')
        self.ax_decision.set_title('Decision Boundary')
        
        # Plot loss history
        if self.loss_history:
            self.ax_loss.plot(self.loss_history[:frame+1], 'g-')
            self.ax_loss.set_title('Fitness History')
            self.ax_loss.set_xlabel('Generation')
            self.ax_loss.set_ylabel('Fitness')
        
        plt.tight_layout()
    
    def animate(self, X: np.ndarray, y: np.ndarray):
        """Create animation of training process"""
        anim = FuncAnimation(
            self.fig,
            lambda frame: self.update_plot(frame, X, y),
            frames=len(self.loss_history),
            interval=100,
            repeat=False
        )
        plt.show()

if __name__ == "__main__":
    # Pattern selection
    print("\nAvailable Patterns:")
    patterns = {
        '1': ('spiral', 'Two interleaving spirals (challenging)'),
        '2': ('circles', 'Concentric circles (tests radial separation)'),
        '3': ('moons', 'Two half-moons (tests curved boundaries)'),
        '4': ('xor', 'XOR pattern (tests logical operations)'),
        '5': ('squares', 'Nested squares (tests straight boundaries)'),
        '6': ('gaussian', 'Gaussian clusters (tests basic clustering)')
    }
    
    print("\nSelect a pattern to learn:")
    for key, (name, desc) in patterns.items():
        print(f"{key}: {name} - {desc}")
    
    pattern_choice = input("\nEnter pattern number (default: 1): ").strip()
    pattern = patterns.get(pattern_choice, patterns['1'])[0]
    
    # Parameter configuration
    print("\nNeural Network and Genetic Algorithm Configuration")
    print("================================================")
    
    # Network architecture
    print("\nNetwork Architecture:")
    print("Current default: [2, 5, 5, 1] (2 inputs, two hidden layers of 5 neurons, 1 output)")
    architecture_input = input("Enter layer sizes separated by spaces (e.g., '2 8 8 1'): ").strip()
    if architecture_input:
        layer_sizes = [int(x) for x in architecture_input.split()]
    else:
        layer_sizes = [2, 5, 5, 1]
    
    # Genetic algorithm parameters
    print("\nGenetic Algorithm Parameters:")
    pop_size = input("Population size (default 50): ").strip()
    population_size = int(pop_size) if pop_size else 50
    
    gen_count = input("Number of generations (default 100): ").strip()
    generations = int(gen_count) if gen_count else 100
    
    # Dataset parameters
    data_size = input("Number of training points (default 200): ").strip()
    n_samples = int(data_size) if data_size else 200
    
    # Create summary
    print("\nRunning with parameters:")
    print(f"Pattern: {pattern}")
    print(f"Network architecture: {layer_sizes}")
    print(f"Population size: {population_size}")
    print(f"Generations: {generations}")
    print(f"Training points: {n_samples}")
    input("\nPress Enter to start training...")
    
    # Create and train network
    network = NeuralNetVisualizer(layer_sizes, population_size=population_size)
    X, y = network.generate_data(n_samples, pattern=pattern)
    
    print("\nTraining network using genetic algorithm...")
    network.genetic_train(X, y, generations=generations)
    
    print("\nTraining complete! Displaying animation...")
    network.animate(X, y)
    
    # Final results
    print(f"\nFinal fitness achieved: {network.best_fitness:.4f}")
    if network.best_fitness >= 0.95:
        print("Excellent performance!")
    elif network.best_fitness >= 0.85:
        print("Good performance!")
    elif network.best_fitness >= 0.75:
        print("Moderate performance. Try increasing network size or generations.")
    else:
        print("Limited performance. Consider:")
        print("- Increasing network size (add more neurons/layers)")
        print("- Running for more generations")
        print("- Increasing population size")
