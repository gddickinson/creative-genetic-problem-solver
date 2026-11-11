"""End-of-run analysis printed by vehicle-routing-optimizer.py (split out to keep that script under 500 lines)."""


def print_summary(opt):
    """Print distance, vehicle utilisation, time-window compliance and suggestions for opt.best_solution."""
    # Print final analysis
    print("\nOptimization Complete!")
    print("=====================")

    print(f"\nFinal Solution Statistics:")
    distance, _ = opt.calculate_fitness(opt.best_solution)
    print(f"Total Distance: {distance:.1f}")
    print(f"Undelivered Locations: {opt.undelivered(opt.best_solution)}")

    # Analyze vehicle utilization
    print("\nVehicle Utilization:")
    for vehicle in opt.best_solution:
        load = sum(opt.locations[i].demand for i in vehicle.route)
        capacity_used = load / vehicle.capacity * 100
        n_deliveries = len(vehicle.route)
        print(f"\nVehicle {vehicle.id}:")
        print(f"  Deliveries: {n_deliveries}")
        print(f"  Capacity Used: {capacity_used:.1f}%")
        print(f"  Route: Depot -> {' -> '.join(map(str, vehicle.route))} -> Depot")

    # Time window analysis
    on_time = 0
    total_deliveries = 0
    for vehicle in opt.best_solution:
        current_time = 0
        prev_loc = opt.depot
        for loc_id in vehicle.route:
            loc = opt.locations[loc_id]
            travel_time = opt.distance(prev_loc, loc) / opt.speed
            current_time += travel_time
            if loc.time_window[0] <= current_time <= loc.time_window[1]:
                on_time += 1
            current_time += loc.service_time
            prev_loc = loc
            total_deliveries += 1

    print(f"\nTime Window Compliance:")
    print(f"On-time Deliveries: {on_time}/{total_deliveries}")
    on_time_share = on_time / total_deliveries if total_deliveries else 0.0
    print(f"On-time Percentage: {on_time_share*100:.1f}%")

    # Suggestions
    print("\nSuggestions for Improvement:")
    if total_deliveries == 0:
        print("- No location could be served within its time window (try a higher vehicle speed)")
    elif on_time_share < 0.9:
        print("- Consider adjusting routes to better meet time windows")

    min_utilization = min(sum(opt.locations[i].demand for i in v.route)/v.capacity
                        for v in opt.best_solution)
    if min_utilization < 0.5:
        print("- Some vehicles are underutilized, consider reducing fleet size")

    max_route = max(len(v.route) for v in opt.best_solution)
    min_route = min(len(v.route) for v in opt.best_solution)
    if max_route > 2 * min_route:
        print("- Route lengths are unbalanced, consider redistributing deliveries")
