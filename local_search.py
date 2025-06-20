import random
import math
import matplotlib.pyplot as plt
from env import BostonTrafficEnv
from local_search_helpers import stop_placement, generate_routes, score_calculator


def local_search(env: BostonTrafficEnv,
                 start,
                 goal,
                 num_stops,
                 num_iterations=500,
                 initial_temp=95.0,
                 cooling_rate=0.98,
                return_history=False):
    """
    Simulated-annealing local search using env for routing and the helper functions.
    Returns best_stops, best_route (list of node tuples), and best_score.
    """
    current_stops = stop_placement(env, start, goal, num_stops)
    current_route = generate_routes(current_stops, env.G)
    current_score = score_calculator(current_route, env.G)

    best_stops, best_route, best_score = list(current_stops), current_route, current_score
    T = initial_temp
    score_history = [current_score] if return_history else None

    for _ in range(num_iterations):
        candidate_stops = stop_placement(env, start, goal, num_stops)
        candidate_route = generate_routes(candidate_stops, env.G)
        candidate_score = score_calculator(candidate_route, env.G)

        delta = candidate_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_stops, current_route, current_score = (
                candidate_stops,
                candidate_route,
                candidate_score
            )
            if current_score < best_score:
                best_stops, best_route, best_score = (
                    list(current_stops),
                    current_route,
                    current_score
                )

        if return_history:
            score_history.append(current_score)
        T *= cooling_rate
        if T < 1e-6:
            break
    if return_history:
        return best_stops, best_route, best_score, score_history
    else:
        return best_stops, best_route, best_score

def main():
    env = BostonTrafficEnv(
        street_csv="boston_street_segments_sam_system.csv",
        traffic_csv="boston_area_with_traffic.csv"
    )

    #Uncomment to use a random start and end goal
    
    start = (-71.05652714499996, 42.36661668700003)
    goal  = (-71.06240483799996, 42.364902536000045)
    stops = [start,(-71.08347291999996, 42.34410348800003),(-71.05378338899999, 42.35742503700004),(-71.10056949699998, 42.34938674000006),
             (-71.07889506799995, 42.35324375000005),(-71.08023887299998, 42.34534430300005),goal]
    """
    start, goal = env.sample_start_goal()
    start_idx = env.nodes.index(start)
    goal_idx = env.nodes.index(goal)
    """
    print(f"Start: {start}, Goal: {goal}")

    best_stops, best_route, best_score = local_search(
        env,
        start,
        goal,
        num_stops=5,
        num_iterations=500,
        initial_temp=0.95,
        cooling_rate=0.98
    )
    print(f"Best score: {best_score:.3f}")
    print(f"Best stops: {best_stops}")
    num_nodes = len(best_route)
    print(f"Number of nodes in best path: {num_nodes}")

    total_traveled = 0.0

    for u, v in zip(best_route[:-1], best_route[1:]):
        edge_data = env.G.get_edge_data(u, v)
        # handle multi-edge dicts by picking the first entry if needed
        if isinstance(edge_data, dict) and 'length' not in edge_data:
            edge_data = list(edge_data.values())[0]
        total_traveled += edge_data['length']
    print(f"Total distance traveled along optimized route: {total_traveled * 100} units")


    plt.ion()
    fig, ax = env.render(path=best_route)

    if best_stops:
        xs, ys = zip(*best_stops)
        ax.scatter(xs, ys, c='red', s=30, marker='o', zorder=5, label='Stops')
        ax.legend(loc='upper right')

    plt.show(block=False)
    for node in best_route:
        x, y = node
        env.agent_dot.set_data([x], [y])
        fig.canvas.draw()
        plt.pause(0.3)

    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
