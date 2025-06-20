import matplotlib.pyplot as plt
from local_search import local_search
from local_search_helpers import *
from traffic_data_Astar import traffic_astar
from env import *

env = BostonTrafficEnv()

# visualize the map with route generated from A* and local search using the same 
# start and goal
def plot_points(ax, start, goal, stops):
        ax.plot(start[0], start[1], marker='o', color='green', markersize=10, label='Start')
        ax.plot(goal[0], goal[1], marker='*', color='red', markersize=12, label='Goal')
        for i, stop in enumerate(stops):
            if stop != start and stop != goal:
                ax.plot(stop[0], stop[1], marker='s', color='orange', markersize=8, label=f'Stop {i}')
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

def compare_route(env):
    start, goal = env.sample_start_goal() 
    stops = stop_placement(env, start, goal, 5)
    
    # A*        
    _, _, astar_path, _, astar_score = traffic_astar(env, start, goal, stops)
    astar_nodes = [item for item in astar_path if isinstance(item, tuple)]
    fig1, ax1 = env.render(path=astar_nodes)
    ax1.set_title(f"A* Route (Score: {astar_score:.2f})")

    plot_points(ax1, start, goal, stops)

    # local saerch
    local_stops, local_route, local_score = local_search(env, start, goal, num_stops=5, num_iterations=500, 
                                               initial_temp=0.95, cooling_rate=0.98)
    local_nodes = [item for item in local_route if isinstance(item, tuple)]
    fig2, ax2 = env.render(path=local_nodes)
    ax2.set_title(f"Local Search Route (Score: {local_score:.2f})")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    plot_points(ax2, start, goal, local_stops)

    fig1.savefig("astar_route.png")
    fig2.savefig("local_route.png")

# compare_route(env)

# visualize the score change in one round of local search
# to observe the decrease in score over time
def local_search_performance(env, num_rounds=5, num_iter=500):
    plt.figure(figsize=(10, 6))
    for i in range(num_rounds):
        start, goal = env.sample_start_goal()
        _, _, _, score_history = local_search(
            env, start, goal, num_stops=5, num_iterations=num_iter,
            initial_temp=0.95, cooling_rate=0.98, return_history=True
        )
        plt.plot(score_history, label=f'Round {i+1}', alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title(f'Local Search Score Progression Over {num_rounds} Rounds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("local_search_performance.png")

# local_search_performance(env, num_rounds=5, num_iter=500)

# run multiple rounds to see which algorithm generally achieves lower score (better)
def multiple_rounds(env, num_round=100):
    result_dict = {'A*':[], 'Local Search':[]}
    for i in range(num_round):
        # generate random start and goals 
        start, goal = env.sample_start_goal()

        # local_search
        best_stops, best_route, best_score = local_search(
            env, start, goal, num_stops=5, num_iterations=500,
            initial_temp=0.95, cooling_rate=0.98
        )
        result_dict['Local Search'].append(best_score)

        # A*
        stops = stop_placement(env, start, goal, num_stops=5)
        best_order, best_dist, full_path, num_nodes, score = traffic_astar(env, start, goal, stops)
        result_dict['A*'].append(score)
        print(i)
    return result_dict

def multiround_result_viz(env, num_rounds=5):
    result_dict = multiple_rounds(env, num_round=num_rounds)
    rounds = list(range(1, num_rounds+1))
    local_scores = result_dict['Local Search']
    astar_scores = result_dict['A*']

    local_avg = sum(local_scores) / len(local_scores)
    astar_avg = sum(astar_scores) / len(astar_scores)

    # generate and save plot
    # x-axis number of round, y-axis score
    # horizontal lines in different color to represent average for two models
    plt.figure(figsize=(10,6))

    plt.plot(rounds, local_scores, label='Local Search', color='red', alpha=0.6)
    plt.plot(rounds, astar_scores, label='A*', color='blue', alpha=0.6)

    plt.axhline(local_avg, color='red', linestyle='--', label=f'Local Search Avg Score: {local_avg:.2f}')
    plt.axhline(astar_avg, color='blue', linestyle='--', label=f'A* Avg Score: {astar_avg:.2f}')

    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.xticks(range(1, num_rounds+1, 10))
    plt.title('Performance Comparison Over Multiple Rounds \n (Lower score indicates better performance)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("multiround_comparison.png")

# multiround_result_viz(env, num_rounds=100)
