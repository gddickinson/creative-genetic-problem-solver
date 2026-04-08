"""Neural network training via genetic algorithm with visualization."""

# Underscore-renamed version of neural-vis.py for valid Python imports.

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
        self.ax_network = plt.subplot2grid((1, 3), (0, 0))
        self.ax_decision = plt.subplot2grid((1, 3), (0, 1))
        self.ax_loss = plt.subplot2grid((1, 3), (0, 2))
        self.loss_history = []
        self.decision_history = []
        self.best_network = None
        self.best_fitness = float('-inf')

    def initialize_network(self) -> List[Layer]:
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            weights = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1
            biases = np.random.randn(1, self.layer_sizes[i+1]) * 0.1
            layers.append(Layer(weights, biases))
        return layers

    def activation(self, x: np.ndarray, func: str = 'relu') -> np.ndarray:
        if func == 'relu':
            return np.maximum(0, x)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        return x

    def forward(self, X: np.ndarray, network: Optional[List[Layer]] = None) -> np.ndarray:
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
        points_per_class = n_samples // 2
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples)

        if pattern == 'spiral':
            for i in range(2):
                t = np.linspace(0, 4*np.pi, points_per_class)
                r = t + np.random.randn(points_per_class) * 0.2
                X[i*points_per_class:(i+1)*points_per_class] = np.column_stack([
                    r*np.cos(t), r*np.sin(t)
                ])
                y[i*points_per_class:(i+1)*points_per_class] = i
        elif pattern == 'circles':
            for i in range(2):
                r = i + 1 + np.random.randn(points_per_class) * 0.1
                t = np.linspace(0, 2*np.pi, points_per_class) + np.random.randn(points_per_class) * 0.2
                X[i*points_per_class:(i+1)*points_per_class] = np.column_stack([
                    r*np.cos(t), r*np.sin(t)
                ])
                y[i*points_per_class:(i+1)*points_per_class] = i
        elif pattern == 'moons':
            t = np.linspace(0, np.pi, points_per_class)
            X[:points_per_class] = np.column_stack([
                np.cos(t), np.sin(t)
            ]) + np.random.randn(points_per_class, 2) * 0.1
            X[points_per_class:] = np.column_stack([
                1 - np.cos(t), 0.5 - np.sin(t)
            ]) + np.random.randn(points_per_class, 2) * 0.1
            y[points_per_class:] = 1
        elif pattern == 'xor':
            X[:points_per_class//2] = np.random.randn(points_per_class//2, 2) * 0.2 + [-1, -1]
            X[points_per_class//2:points_per_class] = np.random.randn(points_per_class//2, 2) * 0.2 + [1, 1]
            X[points_per_class:points_per_class+points_per_class//2] = np.random.randn(points_per_class//2, 2) * 0.2 + [-1, 1]
            X[points_per_class+points_per_class//2:] = np.random.randn(points_per_class//2, 2) * 0.2 + [1, -1]
            y[:points_per_class] = 0
            y[points_per_class:] = 1
        elif pattern == 'gaussian':
            centers = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
            for i in range(2):
                X[i*points_per_class:(i+1)*points_per_class] = np.random.randn(points_per_class, 2) * 0.2 + centers[i]
                y[i*points_per_class:(i+1)*points_per_class] = i

        X = (X - X.mean(axis=0)) / X.std(axis=0)
        return X, y

    def genetic_train(self, X: np.ndarray, y: np.ndarray, generations: int = 100):
        def create_random_network():
            return [Layer(
                np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.1,
                np.random.randn(1, self.layer_sizes[i+1]) * 0.1
            ) for i in range(len(self.layer_sizes) - 1)]

        def fitness(network):
            predictions = self.forward(X, network)
            accuracy = np.mean((predictions > 0.5).flatten() == y)
            return accuracy

        def crossover(net1, net2):
            child = []
            for layer1, layer2 in zip(net1, net2):
                mask = np.random.random(layer1.weights.shape) < 0.5
                weights = np.where(mask, layer1.weights, layer2.weights)
                mask = np.random.random(layer1.biases.shape) < 0.5
                biases = np.where(mask, layer1.biases, layer2.biases)
                child.append(Layer(weights, biases))
            return child

        def mutate(network, rate=0.1):
            mutated = []
            for layer in network:
                mutation_mask = np.random.random(layer.weights.shape) < 0.1
                mutation = np.random.randn(*layer.weights.shape) * rate
                weights = layer.weights.copy()
                weights[mutation_mask] += mutation[mutation_mask]
                mutation_mask = np.random.random(layer.biases.shape) < 0.1
                mutation = np.random.randn(*layer.biases.shape) * rate
                biases = layer.biases.copy()
                biases[mutation_mask] += mutation[mutation_mask]
                mutated.append(Layer(weights, biases))
            return mutated

        population = [create_random_network() for _ in range(self.population_size)]
        self.loss_history = []
        self.decision_history = []

        for generation in range(generations):
            fitness_scores = [(net, fitness(net)) for net in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            if fitness_scores[0][1] > self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_network = fitness_scores[0][0]
            self.loss_history.append(self.best_fitness)
            self.decision_history.append(self.get_decision_boundary(self.best_network))
            next_population = [net for net, _ in fitness_scores[:2]]
            while len(next_population) < self.population_size:
                parents = random.sample([net for net, _ in fitness_scores[:10]], 2)
                child = crossover(parents[0], parents[1])
                child = mutate(child)
                next_population.append(child)
            population = next_population
            if generation % 10 == 0:
                print(f"Generation {generation}, Best Fitness: {self.best_fitness:.4f}")

    def get_decision_boundary(self, network=None):
        xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.forward(X_grid, network if network else self.network)
        return Z.reshape(xx.shape)

    def update_plot(self, frame, X, y):
        self.ax_network.clear()
        self.ax_decision.clear()
        self.ax_loss.clear()
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
        if self.decision_history:
            xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
            self.ax_decision.contourf(xx, yy, self.decision_history[frame], levels=20, cmap='RdBu')
            self.ax_decision.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='black')
        self.ax_decision.set_title('Decision Boundary')
        if self.loss_history:
            self.ax_loss.plot(self.loss_history[:frame+1], 'g-')
            self.ax_loss.set_title('Fitness History')
            self.ax_loss.set_xlabel('Generation')
            self.ax_loss.set_ylabel('Fitness')
        plt.tight_layout()

    def animate(self, X, y):
        anim = FuncAnimation(
            self.fig,
            lambda frame: self.update_plot(frame, X, y),
            frames=len(self.loss_history),
            interval=100,
            repeat=False
        )
        plt.show()

if __name__ == "__main__":
    layer_sizes = [2, 5, 5, 1]
    network = NeuralNetVisualizer(layer_sizes, population_size=50)
    X, y = network.generate_data(200, pattern='spiral')
    print("\nTraining network using genetic algorithm...")
    network.genetic_train(X, y, generations=100)
    print("\nTraining complete! Displaying animation...")
    network.animate(X, y)
    print(f"\nFinal fitness achieved: {network.best_fitness:.4f}")
