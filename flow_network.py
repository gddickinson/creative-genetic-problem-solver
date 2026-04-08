import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Set
import random
from collections import deque
import matplotlib.colors as mcolors

class FlowNetworkVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 7))
        self.pos = None  # Node positions
        self.flows = {}  # Current flows
        self.capacities = {}  # Edge capacities
        self.history = []  # History of flows
        self.max_flow_value = 0
        self.paths = []  # Augmenting paths found

    def generate_network(self, n_nodes: int = 8, edge_density: float = 0.3):
        """Generate a random flow network"""
        # Clear existing graph
        self.graph.clear()
        self.flows = {}
        self.capacities = {}

        # Add nodes
        nodes = list(range(n_nodes))
        self.graph.add_nodes_from(nodes)

        # Identify source (0) and sink (last node)
        source = 0
        sink = n_nodes - 1

        # Generate layered network to ensure path exists
        layers = self.split_into_layers(nodes[1:-1], 3)  # 3 internal layers

        # Add edges between layers
        def add_layer_edges(layer1, layer2):
            for u in layer1:
                # Add edges to random nodes in next layer
                targets = random.sample(layer2,
                                     k=random.randint(1, max(1, len(layer2))))
                for v in targets:
                    capacity = random.randint(5, 20)
                    self.graph.add_edge(u, v)
                    self.capacities[(u, v)] = capacity
                    self.flows[(u, v)] = 0

        # Connect source to first layer
        add_layer_edges([source], layers[0])

        # Connect between internal layers
        for i in range(len(layers)-1):
            add_layer_edges(layers[i], layers[i+1])

        # Connect last layer to sink
        add_layer_edges(layers[-1], [sink])

        # Add some random cross-layer edges
        for _ in range(int(n_nodes * edge_density)):
            u = random.choice(nodes[:-1])  # Any node except sink
            possible_targets = [v for v in nodes if v > u and v != u]
            if possible_targets:
                v = random.choice(possible_targets)
                if not self.graph.has_edge(u, v):
                    capacity = random.randint(5, 20)
                    self.graph.add_edge(u, v)
                    self.capacities[(u, v)] = capacity
                    self.flows[(u, v)] = 0

        # Calculate node positions for visualization
        self.pos = nx.spring_layout(self.graph)

    def split_into_layers(self, nodes: List[int], n_layers: int) -> List[List[int]]:
        """Split nodes into layers"""
        layers = []
        nodes = nodes.copy()
        nodes_per_layer = max(1, len(nodes) // n_layers)

        for _ in range(n_layers-1):
            if nodes:
                layer = random.sample(nodes, k=min(nodes_per_layer, len(nodes)))
                for node in layer:
                    nodes.remove(node)
                layers.append(layer)

        if nodes:
            layers.append(nodes)

        return layers

    def find_augmenting_path(self, source: int, sink: int) -> List[Tuple[int, int]]:
        """Find an augmenting path using BFS"""
        visited = {source: None}
        queue = deque([source])

        while queue:
            u = queue.popleft()
            if u == sink:
                break

            for v in self.graph.neighbors(u):
                residual = self.capacities[(u, v)] - self.flows[(u, v)]
                if v not in visited and residual > 0:
                    visited[v] = u
                    queue.append(v)

        if sink not in visited:
            return []

        # Reconstruct path
        path = []
        current = sink
        while current != source:
            prev = visited[current]
            path.append((prev, current))
            current = prev

        return path[::-1]

    def find_max_flow(self):
        """Find maximum flow using Ford-Fulkerson algorithm"""
        source = 0
        sink = max(self.graph.nodes())
        self.max_flow_value = 0
        self.history = []
        self.paths = []

        # Reset flows
        for edge in self.flows:
            self.flows[edge] = 0

        while True:
            # Find augmenting path
            path = self.find_augmenting_path(source, sink)
            if not path:
                break

            # Find minimum residual capacity along path
            min_residual = min(self.capacities[edge] - self.flows[edge]
                             for edge in path)

            # Update flows along path
            for edge in path:
                self.flows[edge] += min_residual

            self.max_flow_value += min_residual
            self.paths.append(path)
            self.history.append(dict(self.flows))

    def update_plot(self, frame: int):
        """Update visualization for animation"""
        if frame >= len(self.history):
            return

        current_flows = self.history[frame]
        current_path = self.paths[frame] if frame < len(self.paths) else []

        # Clear previous plots
        for ax in self.axes:
            ax.clear()

        # Plot network state
        ax_net = self.axes[0]

        # Draw edges with flows
        edge_colors = []
        edge_widths = []
        for u, v in self.graph.edges():
            flow = current_flows[(u, v)]
            capacity = self.capacities[(u, v)]
            # Color based on flow/capacity ratio
            color = plt.cm.RdYlBu(flow/capacity)
            edge_colors.append(color)
            edge_widths.append(1 + 3 * flow/capacity)

        # Draw the network
        nx.draw_networkx_edges(self.graph, self.pos, ax=ax_net,
                             edge_color=edge_colors, width=edge_widths)

        # Highlight augmenting path
        for edge in current_path:
            nx.draw_networkx_edges(self.graph, self.pos, ax=ax_net,
                                 edgelist=[edge], edge_color='r', width=2)

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, self.pos, ax=ax_net,
                             node_color='lightblue', node_size=500)
        nx.draw_networkx_labels(self.graph, self.pos, ax=ax_net)

        # Add edge labels
        edge_labels = {(u, v): f'{current_flows[(u, v)]}/{self.capacities[(u, v)]}'
                      for u, v in self.graph.edges()}
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels, ax=ax_net)

        ax_net.set_title(f'Network Flow (Step {frame+1})')

        # Plot flow value history
        ax_flow = self.axes[1]
        flow_history = [sum(flows[(0, v)] for v in self.graph.neighbors(0))
                       for flows in self.history[:frame+1]]
        ax_flow.plot(range(len(flow_history)), flow_history, 'b-')
        ax_flow.set_title('Maximum Flow Value Over Time')
        ax_flow.set_xlabel('Step')
        ax_flow.set_ylabel('Flow Value')
        ax_flow.grid(True)

        plt.tight_layout()

    def animate(self, n_nodes: int = 8, edge_density: float = 0.3):
        """Run flow optimization and create animation"""
        self.generate_network(n_nodes, edge_density)
        print("\nGenerating flow network...")
        print(f"Nodes: {n_nodes}")
        print(f"Edges: {self.graph.number_of_edges()}")
        print(f"Edge Density: {edge_density:.2f}")

        print("\nFinding maximum flow...")
        self.find_max_flow()

        print(f"\nMaximum flow found: {self.max_flow_value}")
        print(f"Number of augmenting paths: {len(self.paths)}")

        # Create animation
        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=len(self.history),
            interval=1000,
            repeat=False
        )

        plt.show()

        # Print final analysis
        print("\nNetwork Analysis:")
        print("================")

        # Find bottleneck edges
        bottlenecks = []
        for u, v in self.graph.edges():
            if self.flows[(u, v)] == self.capacities[(u, v)]:
                bottlenecks.append((u, v))

        print(f"\nBottleneck Edges (flow = capacity):")
        for edge in bottlenecks:
            print(f"Edge {edge}: {self.flows[edge]}/{self.capacities[edge]}")

        # Calculate edge utilization statistics
        utilizations = [self.flows[edge]/self.capacities[edge]
                       for edge in self.graph.edges()]
        avg_utilization = np.mean(utilizations) * 100
        print(f"\nAverage Edge Utilization: {avg_utilization:.1f}%")

        # Suggestions for improvement
        print("\nSuggestions for Improvement:")
        if bottlenecks:
            print("- Increase capacity of bottleneck edges")
        if avg_utilization < 30:
            print("- Network may be overprovisioned, consider reducing some capacities")
        elif avg_utilization > 80:
            print("- Network heavily loaded, consider adding alternative paths")

if __name__ == "__main__":
    # Configuration
    print("\nFlow Network Configuration")
    print("=========================")

    n_nodes = input("Number of nodes (default 8): ").strip()
    n_nodes = int(n_nodes) if n_nodes else 8

    density = input("Edge density (0.0-1.0, default 0.3): ").strip()
    density = float(density) if density else 0.3

    # Create and run visualization
    visualizer = FlowNetworkVisualizer()
    visualizer.animate(n_nodes, density)
