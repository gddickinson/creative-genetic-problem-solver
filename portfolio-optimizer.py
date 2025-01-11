import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from typing import List, Tuple, Dict
import random
from dataclasses import dataclass

@dataclass
class Portfolio:
    weights: np.ndarray
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0

class PortfolioOptimizer:
    def __init__(self, n_assets: int = 5, population_size: int = 100):
        self.n_assets = n_assets
        self.population_size = population_size
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
        # Initialize visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.history = []
        self.best_portfolio = None
        self.efficient_frontier = None
        
        # Generate sample data
        self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample return data for assets"""
        print("\nGenerating sample market data...")
        
        # Generate correlated returns
        np.random.seed(42)  # For reproducibility
        
        # Generate random correlation matrix
        corr = np.random.uniform(0.1, 0.7, (self.n_assets, self.n_assets))
        corr = (corr + corr.T) / 2  # Make symmetric
        np.fill_diagonal(corr, 1)    # Diagonal is 1
        
        # Convert correlation to covariance
        volatilities = np.random.uniform(0.1, 0.3, self.n_assets)
        self.cov_matrix = np.outer(volatilities, volatilities) * corr
        
        # Generate expected returns
        self.expected_returns = np.random.uniform(0.05, 0.15, self.n_assets)
        
        # Asset names
        self.asset_names = [f'Asset {i+1}' for i in range(self.n_assets)]
        
        print("\nAsset Characteristics:")
        for i, name in enumerate(self.asset_names):
            print(f"{name}:")
            print(f"  Expected Return: {self.expected_returns[i]*100:.1f}%")
            print(f"  Volatility: {np.sqrt(self.cov_matrix[i,i])*100:.1f}%")
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Portfolio:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        # Expected return
        exp_return = np.sum(self.expected_returns * weights)
        
        # Volatility
        vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        
        # Sharpe ratio
        sharpe = (exp_return - self.risk_free_rate) / vol
        
        return Portfolio(weights, exp_return, vol, sharpe)
    
    def generate_random_portfolio(self) -> np.ndarray:
        """Generate random portfolio weights that sum to 1"""
        weights = np.random.random(self.n_assets)
        return weights / np.sum(weights)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create child portfolio from two parents"""
        child = (parent1 + parent2) / 2
        return child / np.sum(child)
    
    def mutate(self, portfolio: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Randomly mutate portfolio weights"""
        mutation = np.random.normal(0, 0.1, self.n_assets) * (np.random.random(self.n_assets) < mutation_rate)
        mutated = portfolio + mutation
        # Ensure non-negative weights
        mutated = np.maximum(mutated, 0)
        return mutated / np.sum(mutated)
    
    def calculate_efficient_frontier(self, n_points: int = 100):
        """Calculate efficient frontier points"""
        min_ret = min(self.expected_returns)
        max_ret = max(self.expected_returns)
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        efficient_frontier = []
        for target_return in target_returns:
            # Find minimum variance portfolio for this target return
            best_vol = float('inf')
            best_weights = None
            
            for _ in range(1000):
                weights = self.generate_random_portfolio()
                portfolio = self.calculate_portfolio_metrics(weights)
                
                if abs(portfolio.expected_return - target_return) < 0.001 and portfolio.volatility < best_vol:
                    best_vol = portfolio.volatility
                    best_weights = weights
            
            if best_weights is not None:
                efficient_frontier.append((target_return, best_vol))
        
        self.efficient_frontier = np.array(efficient_frontier)
    
    def optimize_portfolio(self, generations: int = 100):
        """Use genetic algorithm to find optimal portfolio"""
        # Initialize population
        population = [self.generate_random_portfolio() for _ in range(self.population_size)]
        best_sharpe = float('-inf')
        
        for generation in range(generations):
            # Evaluate portfolios
            portfolios = [self.calculate_portfolio_metrics(weights) for weights in population]
            portfolios.sort(key=lambda x: x.sharpe_ratio, reverse=True)
            
            # Update best portfolio
            if portfolios[0].sharpe_ratio > best_sharpe:
                best_sharpe = portfolios[0].sharpe_ratio
                self.best_portfolio = portfolios[0]
            
            # Record history
            self.history.append({
                'best_portfolio': portfolios[0],
                'population': portfolios
            })
            
            # Create next generation
            new_population = []
            
            # Elitism - keep best portfolios
            elite_count = self.population_size // 10
            new_population.extend([p.weights for p in portfolios[:elite_count]])
            
            # Create offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament = random.sample(portfolios[:20], 2)
                parent1 = max(tournament, key=lambda x: x.sharpe_ratio).weights
                tournament = random.sample(portfolios[:20], 2)
                parent2 = max(tournament, key=lambda x: x.sharpe_ratio).weights
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
            if generation % 10 == 0:
                print(f"Generation {generation}, Best Sharpe: {best_sharpe:.3f}")
    
    def update_plot(self, frame: int):
        """Update visualization"""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        generation_data = self.history[frame]
        current_portfolios = generation_data['population']
        
        # Plot 1: Risk-Return scatter plot
        ax_scatter = self.axes[0, 0]
        
        # Plot efficient frontier
        if self.efficient_frontier is not None:
            ax_scatter.plot(self.efficient_frontier[:, 1], 
                          self.efficient_frontier[:, 0], 
                          'k--', label='Efficient Frontier')
        
        # Plot current generation portfolios
        vols = [p.volatility for p in current_portfolios]
        rets = [p.expected_return for p in current_portfolios]
        ax_scatter.scatter(vols, rets, c='blue', alpha=0.5, label='Portfolios')
        
        # Plot best portfolio
        best_portfolio = generation_data['best_portfolio']
        ax_scatter.scatter(best_portfolio.volatility, 
                         best_portfolio.expected_return, 
                         c='red', marker='*', s=200, label='Best Portfolio')
        
        ax_scatter.set_xlabel('Volatility')
        ax_scatter.set_ylabel('Expected Return')
        ax_scatter.set_title('Risk-Return Trade-off')
        ax_scatter.legend()
        ax_scatter.grid(True)
        
        # Plot 2: Asset Allocation
        ax_allocation = self.axes[0, 1]
        weights = best_portfolio.weights
        ax_allocation.bar(self.asset_names, weights)
        ax_allocation.set_title('Current Best Portfolio Allocation')
        plt.setp(ax_allocation.xaxis.get_majorticklabels(), rotation=45)
        ax_allocation.set_ylabel('Weight')
        
        # Plot 3: Sharpe Ratio Evolution
        ax_sharpe = self.axes[1, 0]
        sharpe_history = [h['best_portfolio'].sharpe_ratio for h in self.history[:frame+1]]
        ax_sharpe.plot(sharpe_history, 'g-')
        ax_sharpe.set_xlabel('Generation')
        ax_sharpe.set_ylabel('Sharpe Ratio')
        ax_sharpe.set_title('Best Sharpe Ratio Evolution')
        ax_sharpe.grid(True)
        
        # Plot 4: Risk-Return Evolution
        ax_evolution = self.axes[1, 1]
        vol_history = [h['best_portfolio'].volatility for h in self.history[:frame+1]]
        ret_history = [h['best_portfolio'].expected_return for h in self.history[:frame+1]]
        ax_evolution.plot(vol_history, ret_history, 'b.-')
        ax_evolution.set_xlabel('Volatility')
        ax_evolution.set_ylabel('Expected Return')
        ax_evolution.set_title('Best Portfolio Evolution')
        ax_evolution.grid(True)
        
        plt.tight_layout()
    
    def animate(self, generations: int = 100):
        """Run optimization and create animation"""
        print("\nCalculating efficient frontier...")
        self.calculate_efficient_frontier()
        
        print("\nOptimizing portfolio...")
        self.optimize_portfolio(generations)
        
        print("\nCreating visualization...")
        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=len(self.history),
            interval=100,
            repeat=False
        )
        
        plt.show()
        
        # Print final analysis
        print("\nOptimization Complete!")
        print("=====================")
        
        if self.best_portfolio:
            print("\nOptimal Portfolio:")
            print(f"Expected Return: {self.best_portfolio.expected_return*100:.1f}%")
            print(f"Volatility: {self.best_portfolio.volatility*100:.1f}%")
            print(f"Sharpe Ratio: {self.best_portfolio.sharpe_ratio:.3f}")
            
            print("\nOptimal Asset Allocation:")
            for asset, weight in zip(self.asset_names, self.best_portfolio.weights):
                print(f"{asset}: {weight*100:.1f}%")
            
            # Portfolio analysis
            print("\nPortfolio Analysis:")
            
            # Diversification
            herfindahl = np.sum(self.best_portfolio.weights**2)
            print(f"Diversification Score: {(1-herfindahl)*100:.1f}%")
            
            # Risk concentration
            max_weight = np.max(self.best_portfolio.weights)
            print(f"Maximum Single Asset Exposure: {max_weight*100:.1f}%")
            
            # Suggestions
            print("\nSuggestions:")
            if herfindahl > 0.3:
                print("- Consider increasing diversification")
            if max_weight > 0.4:
                print("- Consider reducing maximum position size")
            if self.best_portfolio.volatility > 0.25:
                print("- Portfolio has high volatility, consider risk reduction")

if __name__ == "__main__":
    # Configuration
    print("\nPortfolio Optimization Configuration")
    print("==================================")
    
    n_assets = input("Number of assets (default: 5): ").strip()
    n_assets = int(n_assets) if n_assets else 5
    
    population = input("Population size (default: 100): ").strip()
    population_size = int(population) if population else 100
    
    generations = input("Number of generations (default: 100): ").strip()
    n_generations = int(generations) if generations else 100
    
    # Create and run optimizer
    optimizer = PortfolioOptimizer(n_assets, population_size)
    optimizer.animate(n_generations)