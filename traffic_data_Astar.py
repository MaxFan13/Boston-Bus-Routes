import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
from env import BostonTrafficEnv
import random
import itertools
from local_search_helpers import stop_placement, generate_routes, score_calculator
from a_star import *

env = BostonTrafficEnv(
    street_csv="boston_street_segments_sam_system.csv",
    traffic_csv="boston_area_with_traffic.csv"
)

start = (-71.05652714499996, 42.36661668700003)
goal  = (-71.06240483799996, 42.364902536000045)
stops = [(-71.05652714499996, 42.36661668700003), (-71.07889506799995, 42.35324375000005), (-71.09842382299996, 42.34075869800006), (-71.07383551399994, 42.35285836600008), (-71.05743745899997, 42.35064889500006), (-71.08620486499996, 42.34990945800007), (-71.06240483799996, 42.364902536000045)]
print(f"Original stops order: {stops}")

#Sort intermediate stops to minimize total network distance between consecutive stops
intermediate = stops[1:-1]
best_order = None
best_dist = float('inf')
for perm in itertools.permutations(intermediate):
    seq = [start] + list(perm) + [goal]
    dist = 0
    for a, b in zip(seq, seq[1:]):
        dist += env.astar_path_length(a, b)
    if dist < best_dist:
        best_dist = dist
        best_order = seq
stops = best_order
print(f"Reordered stops for minimal distance (≈ {best_dist * 100} units):")
print(stops)

full_path = []
for i in range(len(stops) - 1):
    seg_start = stops[i]
    seg_goal  = stops[i + 1]
    segment = env.astar_path(seg_start, seg_goal)
    # avoid duplication of connecting node
    if i > 0:
        segment = segment[1:]
    full_path.extend(segment)

num_nodes = len(full_path)
print(f"Total nodes visited: {num_nodes}")
score = score_calculator(full_path, env.G)
print(f"Score: {score}")

#Render map with traffic weights and overlay the path
plt.ion()
fig, ax = env.render(path=full_path)
plt.show(block=False)

#Plot stops as red dots
xs, ys = zip(*stops)
ax.scatter(xs, ys, c='red', s=50, marker='o', zorder=5, label='Stops')
ax.legend(loc='upper right')


#Animating the agent
agent_dot, = ax.plot([], [], 'ro', markersize=8)
for node in full_path:
    x, y = node
    agent_dot.set_data([x], [y])
    fig.canvas.draw()
    plt.pause(0.3)

plt.ioff()
plt.show()
env.close()
