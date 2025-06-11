import random
import math
from env import BostonTrafficEnv

# Assume these functions are implemented elsewhere in your project:
#   stop_placement(start, goal, num_stops) -> list of stops (including start & goal)
#   generate_routes(stops) -> list of nodes/edges representing full movement sequence
#   score_calculator(route) -> numeric score (lower is better)
from your_module import stop_placement, generate_routes, score_calculator


def local_search(env: BostonTrafficEnv,
                 start,
                 goal,
                 num_stops: int,
                 num_iterations: int = 500,
                 initial_temp: float = 100.0,
                 cooling_rate: float = 0.98):
    """
    Simulated-annealing local search to place num_stops stops between start & goal.

    Args:
        env: BostonTrafficEnv instance for routing
        start, goal: start and goal node tuples
        num_stops: number of intermediate stops to generate
        num_iterations: total SA iterations
        initial_temp: starting temperature
        cooling_rate: multiplicative cooling rate per iteration
    Returns:
        best_stops: list of node tuples (including start & goal)
        best_route: list of nodes/edges for the best route
        best_score: score of the best route
    """
    # Initialize with a random placement
    current_stops = stop_placement(start, goal, num_stops)
    current_route = generate_routes(current_stops)
    current_score = score_calculator(current_route)

    best_stops, best_route, best_score = list(current_stops), current_route, current_score
    T = initial_temp

    for i in range(num_iterations):
        # Propose a neighbor: new random stop placement
        candidate_stops = stop_placement(start, goal, num_stops)
        candidate_route = generate_routes(candidate_stops)
        candidate_score = score_calculator(candidate_route)

        delta = candidate_score - current_score
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_stops, current_route, current_score = (
                candidate_stops, candidate_route, candidate_score
            )
            # Update global best
            if current_score < best_score:
                best_stops, best_route, best_score = (
                    list(current_stops), current_route, current_score
                )
        # Cool down
        T *= cooling_rate
        if T < 1e-6:
            break

    return best_stops, best_route, best_score


def main():
    # Create environment
    env = BostonTrafficEnv(
        street_csv="boston_street_segments_sam_system.csv",
        traffic_csv="boston_area_with_traffic.csv"
    )
    # Sample a valid pair
    start, goal = env.sample_start_goal()
    print(f"Start: {start}, Goal: {goal}")

    # Run local search
    best_stops, best_route, best_score = local_search(
        env, start, goal, num_stops=5,
        num_iterations=1000,
        initial_temp=50.0,
        cooling_rate=0.999
    )

    print("Best stops sequence:", best_stops)
    print(f"Best score: {best_score:.3f}")
    # Optionally visualize the best route
    fig, ax = env.render(path=best_route)
    env.agent_dot.set_data([best_route[0][0]], [best_route[0][1]])  # initial dot
    # Draw the route line
    xs, ys = zip(*best_route)
    ax.plot(xs, ys, linestyle='--', color='blue', linewidth=2)
    plt.show()


if __name__ == "__main__":
    main()
