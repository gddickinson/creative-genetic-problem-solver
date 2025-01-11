import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random
import time

@dataclass
class Location:
    id: int
    x: float
    y: float
    demand: float
    time_window: Tuple[float, float]  # (earliest, latest)
    service_time: float
    priority: int  # 1 (low) to 3 (high)

@dataclass
class Vehicle:
    id: int
    capacity: float
    route: List[int]  # List of location IDs
    color: str

class VehicleRoutingOptimizer:
    def __init__(self, n_locations: int = 20, n_vehicles: int = 3):
        self.n_locations = n_locations
        self.n_vehicles = n_vehicles
        self.depot = Location(0, 50, 50, 0, (0, 24), 0, 1)  # Central depot
        self.locations = [self.depot]
        self.vehicles = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

        # Setup visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))

        # Generate problem data
        self.generate_problem()

    def generate_problem(self):
        """Generate random problem instance"""
        print("\nGenerating delivery locations and constraints...")

        # Generate delivery locations
        for i in range(1, self.n_locations):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            demand = random.uniform(1, 10)
            # Time windows within 24-hour period
            window_start = random.uniform(0, 12)
            window_length = random.uniform(4, 8)
            time_window = (window_start, window_start + window_length)
            service_time = random.uniform(0.2, 1.0)
            priority = random.randint(1, 3)

            self.locations.append(Location(i, x, y, demand, time_window,
                                        service_time, priority))

        # Generate vehicles with different capacities
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i in range(self.n_vehicles):
            capacity = random.uniform(30, 50)
            self.vehicles.append(Vehicle(i, capacity, [], colors[i % len(colors)]))

        print(f"\nProblem Configuration:")
        print(f"Locations: {self.n_locations}")
        print(f"Vehicles: {self.n_vehicles}")

        # Print location details
        print("\nLocation Details:")
        for loc in self.locations[1:]:  # Skip depot
            print(f"Location {loc.id}:")
            print(f"  Demand: {loc.demand:.1f}")
            print(f"  Time Window: {loc.time_window[0]:.1f}-{loc.time_window[1]:.1f}")
            print(f"  Priority: {loc.priority}")

    def distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate Euclidean distance between locations"""
        return np.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)

    def calculate_route_metrics(self, vehicle: Vehicle) -> Tuple[float, bool]:
        """Calculate route length and check constraints"""
        if not vehicle.route:
            return 0, True

        total_distance = 0
        current_time = 0
        current_load = 0
        valid = True

        prev_loc = self.depot
        for loc_id in vehicle.route:
            loc = self.locations[loc_id]

            # Add travel time (assume 1 unit distance = 1 unit time)
            travel_time = self.distance(prev_loc, loc)
            total_distance += travel_time
            current_time += travel_time

            # Check time window
            if current_time < loc.time_window[0]:
                current_time = loc.time_window[0]
            elif current_time > loc.time_window[1]:
                valid = False

            # Add service time
            current_time += loc.service_time

            # Check capacity
            current_load += loc.demand
            if current_load > vehicle.capacity:
                valid = False

            prev_loc = loc

        # Return to depot
        total_distance += self.distance(prev_loc, self.depot)

        return total_distance, valid

    def create_initial_solution(self) -> List[Vehicle]:
        """Create initial solution using nearest neighbor with constraints"""
        vehicles = [Vehicle(v.id, v.capacity, [], v.color) for v in self.vehicles]
        unassigned = list(range(1, self.n_locations))

        # Sort locations by priority (high to low)
        unassigned.sort(key=lambda x: -self.locations[x].priority)

        for loc_id in unassigned:
            best_vehicle = None
            best_position = None
            best_cost = float('inf')

            for vehicle in vehicles:
                # Try inserting location at each position in route
                for i in range(len(vehicle.route) + 1):
                    # Create new route with location inserted
                    new_route = vehicle.route.copy()
                    new_route.insert(i, loc_id)
                    vehicle.route = new_route

                    # Check if route is valid and calculate cost
                    distance, valid = self.calculate_route_metrics(vehicle)
                    if valid and distance < best_cost:
                        best_vehicle = vehicle
                        best_position = i
                        best_cost = distance

                    # Restore original route
                    vehicle.route = vehicle.route[:-1]

            if best_vehicle is not None:
                best_vehicle.route.insert(best_position, loc_id)
            else:
                print(f"Warning: Could not assign location {loc_id}")

        return vehicles

    def crossover(self, parent1: List[Vehicle], parent2: List[Vehicle]) -> List[Vehicle]:
        """Create child solution from two parents"""
        child = [Vehicle(v.id, v.capacity, [], v.color) for v in self.vehicles]

        # Get all locations from parent1
        all_locations = []
        for vehicle in parent1:
            all_locations.extend(vehicle.route)

        # Randomly select crossover point
        crossover_point = random.randint(0, len(all_locations))

        # Take first part from parent1
        locations_taken = set()
        for i, vehicle in enumerate(parent1):
            for loc_id in vehicle.route[:crossover_point]:
                child[i].route.append(loc_id)
                locations_taken.add(loc_id)

        # Take remaining locations in order from parent2
        current_vehicle = 0
        for vehicle in parent2:
            for loc_id in vehicle.route:
                if loc_id not in locations_taken:
                    child[current_vehicle].route.append(loc_id)
                    current_vehicle = (current_vehicle + 1) % len(child)

        return child

    def mutate(self, solution: List[Vehicle], mutation_rate: float = 0.1) -> List[Vehicle]:
        """Randomly modify solution"""
        if random.random() > mutation_rate:
            return solution

        mutated = [Vehicle(v.id, v.capacity, v.route.copy(), v.color)
                  for v in solution]

        # Apply random mutation
        mutation_type = random.choice(['swap', 'reverse', 'relocate'])

        if mutation_type == 'swap':
            # Swap two random locations
            v1 = random.randint(0, len(mutated)-1)
            v2 = random.randint(0, len(mutated)-1)
            if mutated[v1].route and mutated[v2].route:
                i1 = random.randint(0, len(mutated[v1].route)-1)
                i2 = random.randint(0, len(mutated[v2].route)-1)
                mutated[v1].route[i1], mutated[v2].route[i2] = \
                    mutated[v2].route[i2], mutated[v1].route[i1]

        elif mutation_type == 'reverse':
            # Reverse part of a route
            v = random.randint(0, len(mutated)-1)
            if len(mutated[v].route) > 2:
                i = random.randint(0, len(mutated[v].route)-2)
                j = random.randint(i+1, len(mutated[v].route))
                mutated[v].route[i:j] = reversed(mutated[v].route[i:j])

        else:  # relocate
            # Move location to different position or vehicle
            v1 = random.randint(0, len(mutated)-1)
            if mutated[v1].route:
                i = random.randint(0, len(mutated[v1].route)-1)
                v2 = random.randint(0, len(mutated)-1)
                if v1 != v2 or len(mutated[v1].route) > 1:
                    loc = mutated[v1].route.pop(i)
                    j = random.randint(0, len(mutated[v2].route))
                    mutated[v2].route.insert(j, loc)

        return mutated

    def calculate_fitness(self, solution: List[Vehicle]) -> Tuple[float, bool]:
        """Calculate total cost and validity of solution"""
        total_distance = 0
        valid = True

        for vehicle in solution:
            distance, route_valid = self.calculate_route_metrics(vehicle)
            total_distance += distance
            valid = valid and route_valid

        return total_distance, valid

    def optimize(self, generations: int = 100, population_size: int = 50):
        """Run genetic algorithm to optimize routes"""
        # Initialize population
        population = []
        for _ in range(population_size):
            solution = self.create_initial_solution()
            population.append(solution)

        for generation in range(generations):
            # Evaluate solutions
            fitness_scores = []
            for solution in population:
                distance, valid = self.calculate_fitness(solution)
                fitness = distance if valid else float('inf')
                fitness_scores.append((solution, fitness))

            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1])

            # Update best solution
            if fitness_scores[0][1] < self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_solution = [Vehicle(v.id, v.capacity, v.route.copy(), v.color)
                                    for v in fitness_scores[0][0]]

            # Record history
            self.history.append({
                'best_solution': fitness_scores[0][0],
                'best_fitness': fitness_scores[0][1],
                'population': [s for s, _ in fitness_scores]
            })

            if generation % 10 == 0:
                print(f"Generation {generation}, Best Distance: {self.best_fitness:.1f}")

            # Create next generation
            new_population = []

            # Elitism
            elite_count = population_size // 10
            new_population.extend([s for s, _ in fitness_scores[:elite_count]])

            # Create offspring
            while len(new_population) < population_size:
                # Tournament selection
                tournament = random.sample(fitness_scores[:20], 2)
                parent1 = min(tournament, key=lambda x: x[1])[0]
                tournament = random.sample(fitness_scores[:20], 2)
                parent2 = min(tournament, key=lambda x: x[1])[0]

                # Create and mutate child
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

    def update_plot(self, frame: int):
        """Update visualization"""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        solution = self.history[frame]['best_solution']
        fitness = self.history[frame]['best_fitness']

        # Plot 1: Current routes
        ax_routes = self.axes[0, 0]

        # Plot locations
        for loc in self.locations[1:]:  # Skip depot
            color = 'gray'
            if loc.priority == 3:
                color = 'red'
            elif loc.priority == 2:
                color = 'orange'
            ax_routes.scatter(loc.x, loc.y, c=color, s=100)
            ax_routes.annotate(f'{loc.id}', (loc.x, loc.y))

        # Plot depot
        ax_routes.scatter(self.depot.x, self.depot.y, c='black',
                         marker='s', s=100, label='Depot')

        # Plot routes
        for vehicle in solution:
            if vehicle.route:
                route = [self.depot] + [self.locations[i] for i in vehicle.route] + [self.depot]
                xs = [loc.x for loc in route]
                ys = [loc.y for loc in route]
                ax_routes.plot(xs, ys, c=vehicle.color,
                             label=f'Vehicle {vehicle.id}')

        ax_routes.set_title(f'Current Routes (Distance: {fitness:.1f})')
        ax_routes.legend()
        ax_routes.grid(True)

        # Plot 2: Vehicle loads
        ax_loads = self.axes[0, 1]
        for vehicle in solution:
            total_load = sum(self.locations[i].demand for i in vehicle.route)
            capacity_used = total_load / vehicle.capacity * 100
            ax_loads.bar(f'Vehicle {vehicle.id}', capacity_used,
                        color=vehicle.color)
        ax_loads.set_title('Vehicle Capacity Utilization')
        ax_loads.set_ylabel('Capacity Used (%)')
        ax_loads.grid(True)

        # Plot 3: Route length evolution
        ax_evolution = self.axes[1, 0]
        history_fitness = [h['best_fitness'] for h in self.history[:frame+1]]
        ax_evolution.plot(history_fitness, 'b-')
        ax_evolution.set_title('Best Route Length Evolution')
        ax_evolution.set_xlabel('Generation')
        ax_evolution.set_ylabel('Total Distance')
        ax_evolution.grid(True)

        # Plot 4: Time windows
        ax_time = self.axes[1, 1]
        current_vehicle = 0
        current_time = 0

        for vehicle in solution:
            prev_loc = self.depot
            for loc_id in vehicle.route:
                loc = self.locations[loc_id]
                # Calculate arrival time
                travel_time = self.distance(prev_loc, loc)
                current_time += travel_time

                # Plot time window
                window_start, window_end = loc.time_window
                ax_time.barh(loc_id, window_end - window_start,
                           left=window_start, alpha=0.3,
                           color=vehicle.color)

                # Plot actual service time
                ax_time.plot(current_time, loc_id, 'o',
                           color=vehicle.color)

                # Add service time
                current_time += loc.service_time
                prev_loc = loc

            # Reset for next vehicle
            current_time = 0
            current_vehicle += 1

        ax_time.set_title('Delivery Time Windows')
        ax_time.set_xlabel('Time of Day')
        ax_time.set_ylabel('Location ID')
        ax_time.grid(True)

        plt.tight_layout()

    def animate(self, generations: int = 100, population_size: int = 50):
        """Run optimization and create animation"""
        print("\nOptimizing routes...")
        self.optimize(generations, population_size)

        print("\nCreating visualization...")
        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=len(self.history),
            interval=200,
            repeat=False
        )

        plt.show()

        # Print final analysis
        print("\nOptimization Complete!")
        print("=====================")

        print(f"\nFinal Solution Statistics:")
        print(f"Total Distance: {self.best_fitness:.1f}")

        # Analyze vehicle utilization
        print("\nVehicle Utilization:")
        for vehicle in self.best_solution:
            load = sum(self.locations[i].demand for i in vehicle.route)
            capacity_used = load / vehicle.capacity * 100
            n_deliveries = len(vehicle.route)
            print(f"\nVehicle {vehicle.id}:")
            print(f"  Deliveries: {n_deliveries}")
            print(f"  Capacity Used: {capacity_used:.1f}%")
            print(f"  Route: Depot -> {' -> '.join(map(str, vehicle.route))} -> Depot")

        # Time window analysis
        on_time = 0
        total_deliveries = 0
        for vehicle in self.best_solution:
            current_time = 0
            prev_loc = self.depot
            for loc_id in vehicle.route:
                loc = self.locations[loc_id]
                travel_time = self.distance(prev_loc, loc)
                current_time += travel_time
                if loc.time_window[0] <= current_time <= loc.time_window[1]:
                    on_time += 1
                current_time += loc.service_time
                prev_loc = loc
                total_deliveries += 1

        print(f"\nTime Window Compliance:")
        print(f"On-time Deliveries: {on_time}/{total_deliveries}")
        print(f"On-time Percentage: {(on_time/total_deliveries*100):.1f}%")

        # Suggestions
        print("\nSuggestions for Improvement:")
        if on_time/total_deliveries < 0.9:
            print("- Consider adjusting routes to better meet time windows")

        min_utilization = min(sum(self.locations[i].demand for i in v.route)/v.capacity
                            for v in self.best_solution)
        if min_utilization < 0.5:
            print("- Some vehicles are underutilized, consider reducing fleet size")

        max_route = max(len(v.route) for v in self.best_solution)
        min_route = min(len(v.route) for v in self.best_solution)
        if max_route > 2 * min_route:
            print("- Route lengths are unbalanced, consider redistributing deliveries")

if __name__ == "__main__":
    # Configuration
    print("\nVehicle Routing Configuration")
    print("============================")

    n_locations = input("Number of delivery locations (default: 20): ").strip()
    n_locations = int(n_locations) if n_locations else 20

    n_vehicles = input("Number of vehicles (default: 3): ").strip()
    n_vehicles = int(n_vehicles) if n_vehicles else 3

    population = input("Population size (default: 50): ").strip()
    population_size = int(population) if population else 50

    generations = input("Number of generations (default: 100): ").strip()
    n_generations = int(generations) if generations else 100

    # Create and run optimizer
    optimizer = VehicleRoutingOptimizer(n_locations, n_vehicles)
    optimizer.animate(n_generations, population_size)
