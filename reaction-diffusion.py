import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from typing import Tuple, List
import random

class ReactionDiffusionSimulator:
    def __init__(self, width: int = 100, height: int = 100):
        self.width = width
        self.height = height
        self.Du = 0.16    # Diffusion rate of U
        self.Dv = 0.08    # Diffusion rate of V
        self.F = 0.035    # Feed rate
        self.k = 0.065    # Kill rate
        self.dt = 1.0     # Time step
        self.steps_per_frame = 10
        
        # Initialize concentrations
        self.U = np.ones((height, width))
        self.V = np.zeros((height, width))
        
        # Setup visualization
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 7))
        self.history = []
        self.pattern_metrics = []
        
    def initialize_pattern(self, pattern_type: str = 'random'):
        """Initialize starting pattern"""
        if pattern_type == 'random':
            # Random spots
            for _ in range(10):
                x = random.randint(45, 55)
                y = random.randint(45, 55)
                self.V[y-3:y+3, x-3:x+3] = 1.0
                
        elif pattern_type == 'center':
            # Single central spot
            cx, cy = self.width//2, self.height//2
            self.V[cy-3:cy+3, cx-3:cx+3] = 1.0
            
        elif pattern_type == 'stripes':
            # Vertical stripes
            for x in range(0, self.width, 10):
                self.V[:, x:x+3] = 1.0
                
        elif pattern_type == 'checkerboard':
            # Checkerboard pattern
            for i in range(0, self.height, 10):
                for j in range(0, self.width, 10):
                    self.V[i:i+5, j:j+5] = 1.0
    
    def set_parameters(self, pattern_name: str):
        """Set parameters for different pattern types"""
        if pattern_name == 'spots':
            self.F = 0.035
            self.k = 0.065
        elif pattern_name == 'stripes':
            self.F = 0.030
            self.k = 0.060
        elif pattern_name == 'waves':
            self.F = 0.025
            self.k = 0.055
        elif pattern_name == 'chaos':
            self.F = 0.020
            self.k = 0.050
        
        print(f"\nParameters set for {pattern_name}:")
        print(f"Feed rate (F): {self.F}")
        print(f"Kill rate (k): {self.k}")
        
    def laplacian(self, X: np.ndarray) -> np.ndarray:
        """Calculate Laplacian using periodic boundary conditions"""
        top = np.roll(X, 1, axis=0)
        bottom = np.roll(X, -1, axis=0)
        left = np.roll(X, 1, axis=1)
        right = np.roll(X, -1, axis=1)
        return (top + bottom + left + right - 4 * X)
    
    def step(self):
        """Perform one time step of the simulation"""
        Lu = self.laplacian(self.U)
        Lv = self.laplacian(self.V)
        
        UVV = self.U * self.V * self.V
        
        dU = self.Du * Lu - UVV + self.F * (1.0 - self.U)
        dV = self.Dv * Lv + UVV - (self.F + self.k) * self.V
        
        self.U += dU * self.dt
        self.V += dV * self.dt
        
        # Ensure values stay in valid range
        self.U = np.clip(self.U, 0, 1)
        self.V = np.clip(self.V, 0, 1)
    
    def calculate_metrics(self) -> dict:
        """Calculate pattern metrics"""
        # Average concentration
        avg_u = np.mean(self.U)
        avg_v = np.mean(self.V)
        
        # Pattern complexity (using gradient magnitude)
        gy, gx = np.gradient(self.V)
        complexity = np.mean(np.sqrt(gx**2 + gy**2))
        
        # Spot/stripe detection
        threshold = np.mean(self.V) + np.std(self.V)
        features = self.V > threshold
        feature_count = np.sum(features)
        
        return {
            'avg_u': avg_u,
            'avg_v': avg_v,
            'complexity': complexity,
            'feature_count': feature_count
        }
    
    def update_plot(self, frame: int):
        """Update visualization"""
        # Clear previous plots but maintain figure and axes
        self.axes[0].clear()
        self.axes[1].clear()
        
        # Run simulation steps
        for _ in range(self.steps_per_frame):
            self.step()
        
        # Calculate and store metrics
        metrics = self.calculate_metrics()
        self.pattern_metrics.append(metrics)
        
        # Plot current state
        im = self.axes[0].imshow(self.V, cmap='RdGy', interpolation='nearest')
        self.axes[0].set_title(f'Pattern Evolution (Step {frame*self.steps_per_frame})')
        if not hasattr(self, 'colorbar'):
            self.colorbar = plt.colorbar(im, ax=self.axes[0], label='Concentration V')
        
        # Plot metrics
        frames = range(len(self.pattern_metrics))
        complexities = [m['complexity'] for m in self.pattern_metrics]
        feature_counts = [m['feature_count'] for m in self.pattern_metrics]
        
        self.axes[1].plot(frames, complexities, 'b-', label='Pattern Complexity')
        ax_metrics2 = self.axes[1].twinx()
        ax_metrics2.plot(frames, feature_counts, 'r-', label='Feature Count')
        
        self.axes[1].set_xlabel('Time Step')
        self.axes[1].set_ylabel('Pattern Complexity', color='b')
        ax_metrics2.set_ylabel('Feature Count', color='r')
        
        # Handle legend
        lines1, labels1 = self.axes[1].get_legend_handles_labels()
        lines2, labels2 = ax_metrics2.get_legend_handles_labels()
        self.axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Maintain layout
        self.fig.tight_layout()
    
    def animate(self, n_frames: int = 200, pattern_type: str = 'spots', 
                init_pattern: str = 'random'):
        """Run simulation and create animation"""
        # Set parameters
        self.set_parameters(pattern_type)
        
        # Initialize pattern
        self.initialize_pattern(init_pattern)
        
        print("\nStarting simulation...")
        print(f"Grid size: {self.width}x{self.height}")
        print(f"Pattern type: {pattern_type}")
        print(f"Initial condition: {init_pattern}")
        
        # Create animation
        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=n_frames,
            interval=50,
            repeat=False
        )
        
        plt.show()
        
        # Print final analysis
        print("\nSimulation Complete!")
        print("===================")
        
        final_metrics = self.pattern_metrics[-1]
        print(f"\nFinal Pattern Metrics:")
        print(f"Pattern Complexity: {final_metrics['complexity']:.3f}")
        print(f"Feature Count: {int(final_metrics['feature_count'])}")
        
        # Pattern classification
        complexity = final_metrics['complexity']
        if complexity < 0.1:
            print("\nPattern Classification: Uniform/Stable")
        elif complexity < 0.2:
            print("\nPattern Classification: Regular Spots/Stripes")
        elif complexity < 0.3:
            print("\nPattern Classification: Complex Patterns")
        else:
            print("\nPattern Classification: Chaotic/Turbulent")
        
        # Suggestions
        print("\nSuggestions for Exploration:")
        print("- Try different initial conditions")
        print("- Adjust feed rate (F) and kill rate (k)")
        print("- Modify diffusion rates for different scales")
        print("- Experiment with grid size")

if __name__ == "__main__":
    # Configuration
    print("\nReaction-Diffusion Pattern Simulator")
    print("===================================")
    
    # Pattern type selection
    print("\nAvailable Patterns:")
    patterns = {
        '1': ('spots', 'Spotted patterns (like leopard spots)'),
        '2': ('stripes', 'Striped patterns (like zebra stripes)'),
        '3': ('waves', 'Wave-like patterns'),
        '4': ('chaos', 'Chaotic/turbulent patterns')
    }
    
    for key, (name, desc) in patterns.items():
        print(f"{key}: {name} - {desc}")
    
    pattern_choice = input("\nSelect pattern type (default: 1): ").strip()
    pattern_type = patterns.get(pattern_choice, patterns['1'])[0]
    
    # Initial condition selection
    print("\nInitial Conditions:")
    print("1: Random spots")
    print("2: Single central spot")
    print("3: Vertical stripes")
    print("4: Checkerboard")
    
    init_choice = input("\nSelect initial condition (default: 1): ").strip()
    init_patterns = {
        '1': 'random',
        '2': 'center',
        '3': 'stripes',
        '4': 'checkerboard'
    }
    init_pattern = init_patterns.get(init_choice, 'random')
    
    # Grid size
    size = input("\nGrid size (default: 100): ").strip()
    size = int(size) if size else 100
    
    # Number of frames
    frames = input("Number of frames (default: 200): ").strip()
    frames = int(frames) if frames else 200
    
    # Create and run simulator
    simulator = ReactionDiffusionSimulator(size, size)
    simulator.animate(frames, pattern_type, init_pattern)