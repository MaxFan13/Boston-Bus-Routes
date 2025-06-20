import matplotlib.pyplot as plt
from local_search import local_search
from local_search_helpers import *
from traffic_data_Astar import traffic_astar
from env import *

env = BostonTrafficEnv()

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