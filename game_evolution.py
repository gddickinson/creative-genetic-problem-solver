import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import seaborn as sns

@dataclass
class Strategy:
    name: str
    cooperate_prob: float  # Probability of cooperation
    memory_length: int     # How many past moves to remember
    mutation_rate: float   # How likely to change behavior

    def decide(self, history: List[int]) -> bool:
        """Make a decision based on strategy"""
        if not history or len(history) < self.memory_length:
            return random.random() < self.cooperate_prob

        recent_history = history[-self.memory_length:]
        cooperation_rate = sum(recent_history) / len(recent_history)

        # Adjust cooperation probability based on opponent's history
        adjusted_prob = self.cooperate_prob * (1 + cooperation_rate) / 2
        return random.random() < adjusted_prob

class GameEvolutionVisualizer:
    def __init__(self, population_size: int = 100, n_generations: int = 100):
        self.population_size = population_size
        self.n_generations = n_generations
        self.strategies = self.initialize_strategies()
        self.population = []
        self.history = []
        self.payoff_matrix = {}
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))

    def initialize_strategies(self) -> Dict[str, Strategy]:
        """Initialize different strategies"""
        return {
            'Cooperative': Strategy('Cooperative', 0.9, 1, 0.1),
            'Defective': Strategy('Defective', 0.1, 1, 0.1),
            'Tit-for-Tat': Strategy('Tit-for-Tat', 0.5, 1, 0.1),
            'Grudger': Strategy('Grudger', 0.7, 3, 0.1),
            'Random': Strategy('Random', 0.5, 0, 0.1)
        }

    def setup_prisoners_dilemma(self):
        """Setup Prisoner's Dilemma payoff matrix"""
        self.payoff_matrix = {
            (True, True): (3, 3),    # Both cooperate
            (True, False): (0, 5),   # Player 1 cooperates, Player 2 defects
            (False, True): (5, 0),   # Player 1 defects, Player 2 cooperates
            (False, False): (1, 1)   # Both defect
        }

    def setup_hawk_dove(self):
        """Setup Hawk-Dove game payoff matrix"""
        self.payoff_matrix = {
            (True, True): (0, 0),    # Both peaceful (dove)
            (True, False): (1, 3),   # Dove vs Hawk
            (False, True): (3, 1),   # Hawk vs Dove
            (False, False): (-2, -2) # Both aggressive (hawk)
        }

    def setup_stag_hunt(self):
        """Setup Stag Hunt game payoff matrix"""
        self.payoff_matrix = {
            (True, True): (4, 4),    # Both hunt stag
            (True, False): (0, 3),   # Hunt stag vs hunt rabbit
            (False, True): (3, 0),   # Hunt rabbit vs hunt stag
            (False, False): (2, 2)   # Both hunt rabbit
        }

    def play_game(self, strategy1: Strategy, strategy2: Strategy,
                  n_rounds: int = 10) -> Tuple[float, float]:
        """Play a game between two strategies"""
        history1, history2 = [], []
        total_payoff1, total_payoff2 = 0, 0

        for _ in range(n_rounds):
            move1 = strategy1.decide(history2)
            move2 = strategy2.decide(history1)

            payoff1, payoff2 = self.payoff_matrix[(move1, move2)]
            total_payoff1 += payoff1
            total_payoff2 += payoff2

            history1.append(int(move1))
            history2.append(int(move2))

        return total_payoff1 / n_rounds, total_payoff2 / n_rounds

    def evolve_population(self):
        """Evolve population through generations"""
        # Initialize population
        self.population = random.choices(list(self.strategies.values()), k=self.population_size)
        population_history = []
        strategy_performance = []

        for generation in range(self.n_generations):
            # Play games
            payoffs = np.zeros(self.population_size)
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if i != j:
                        payoff1, _ = self.play_game(self.population[i], self.population[j])
                        payoffs[i] += payoff1

            # Ensure non-negative payoffs for reproduction
            payoffs = payoffs - np.min(payoffs)
            payoffs = payoffs + 1e-10

            # Record strategy distribution
            strategy_counts = {}
            for strategy in self.strategies.values():
                count = sum(1 for p in self.population if p.name == strategy.name)
                strategy_counts[strategy.name] = count / self.population_size
            population_history.append(strategy_counts)

            # Record average performance
            performance = {}
            for strategy in self.strategies.values():
                strategy_payoffs = [payoffs[i] for i in range(len(self.population))
                                  if self.population[i].name == strategy.name]
                if strategy_payoffs:
                    performance[strategy.name] = np.mean(strategy_payoffs)
                else:
                    performance[strategy.name] = 0
            strategy_performance.append(performance)

            # Selection and reproduction
            reproduction_probs = payoffs / np.sum(payoffs)

            # Create next generation
            new_population = []
            for _ in range(self.population_size):
                parent = self.population[np.random.choice(len(self.population), p=reproduction_probs)]
                cooperate_prob = parent.cooperate_prob
                if random.random() < parent.mutation_rate:
                    cooperate_prob = np.clip(
                        cooperate_prob + np.random.normal(0, 0.1),
                        0.0, 1.0
                    )
                child = Strategy(
                    parent.name,
                    cooperate_prob,
                    parent.memory_length,
                    parent.mutation_rate
                )
                new_population.append(child)

            self.population = new_population
            self.history.append((population_history[-1], strategy_performance[-1]))

    def update_plot(self, frame: int):
        """Update visualization for animation"""
        if frame >= len(self.history):
            return

        population_dist, performance = self.history[frame]

        for ax in self.axes.flat:
            ax.clear()

        ax_pop = self.axes[0, 0]
        strategies = list(population_dist.keys())
        populations = list(population_dist.values())
        colors = sns.color_palette("husl", len(strategies))
        ax_pop.bar(strategies, populations, color=colors)
        ax_pop.set_title("Strategy Distribution")
        ax_pop.set_ylim(0, 1)
        plt.setp(ax_pop.xaxis.get_majorticklabels(), rotation=45)

        ax_perf = self.axes[0, 1]
        perf_strategies = list(performance.keys())
        scores = list(performance.values())
        ax_perf.bar(perf_strategies, scores, color=colors)
        ax_perf.set_title("Strategy Performance")
        plt.setp(ax_perf.xaxis.get_majorticklabels(), rotation=45)

        ax_hist = self.axes[1, 0]
        for i, strategy_name in enumerate(self.strategies.keys()):
            strategy_history = [gen_hist[0][strategy_name] for gen_hist in self.history[:frame+1]]
            ax_hist.plot(range(len(strategy_history)), strategy_history,
                        label=strategy_name, color=colors[i])

        ax_hist.set_title("Population History")
        ax_hist.set_xlabel("Generation")
        ax_hist.set_ylabel("Population Share")
        ax_hist.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_hist.set_ylim(0, 1)

        ax_fit = self.axes[1, 1]
        avg_fitness = [np.mean(list(perf.values())) for _, perf in self.history[:frame+1]]
        ax_fit.plot(range(len(avg_fitness)), avg_fitness, 'k-')
        ax_fit.set_title("Average Population Fitness")
        ax_fit.set_xlabel("Generation")
        ax_fit.set_ylabel("Fitness")

        plt.tight_layout()

    def animate(self, game_type: str = 'prisoners_dilemma'):
        """Run evolution and create animation"""
        if game_type == 'prisoners_dilemma':
            self.setup_prisoners_dilemma()
            game_name = "Prisoner's Dilemma"
        elif game_type == 'hawk_dove':
            self.setup_hawk_dove()
            game_name = "Hawk-Dove Game"
        else:
            self.setup_stag_hunt()
            game_name = "Stag Hunt"

        print(f"\nSimulating {game_name}...")
        self.evolve_population()

        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=len(self.history),
            interval=100,
            repeat=False
        )

        plt.show()

        print("\nSimulation Complete!")
        print("===================")
        print(f"\nGame Type: {game_name}")
        print(f"Total Generations: {self.n_generations}")

        final_dist = self.history[-1][0]
        print("\nFinal Population Distribution:")
        for strategy, proportion in final_dist.items():
            percentage = proportion * 100
            if percentage >= 1:
                print(f"- {strategy}: {percentage:.1f}%")

        dominant_strategy = max(final_dist.items(), key=lambda x: x[1])[0]
        print(f"\nDominant Strategy: {dominant_strategy}")

if __name__ == "__main__":
    print("\nAvailable Games:")
    games = {
        '1': ('prisoners_dilemma', "Prisoner's Dilemma - Classic cooperation vs. defection"),
        '2': ('hawk_dove', "Hawk-Dove Game - Competition vs. peaceful coexistence"),
        '3': ('stag_hunt', "Stag Hunt - Coordination and trust")
    }

    print("\nSelect a game to simulate:")
    for key, (name, desc) in games.items():
        print(f"{key}: {desc}")

    game_choice = input("\nEnter game number (default: 1): ").strip()
    game_type = games.get(game_choice, games['1'])[0]

    pop_size = input("Population size (default 100): ").strip()
    population_size = int(pop_size) if pop_size else 100

    gen_count = input("Number of generations (default 100): ").strip()
    generations = int(gen_count) if gen_count else 100

    visualizer = GameEvolutionVisualizer(population_size, generations)
    visualizer.animate(game_type)
