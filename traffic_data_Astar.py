import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
from env import BostonTrafficEnv
import random
import itertools

# 1) Initialize environment (loads CSVs, builds graph)
env = BostonTrafficEnv(
    street_csv="boston_street_segments_sam_system.csv",
    traffic_csv="boston_area_with_traffic.csv"
)

# 2) Define start, goal, and intermediate stops
start = (-71.05652714499996, 42.36661668700003)
goal  = (-71.06240483799996, 42.364902536000045)
stops = [
    start,
    (-71.08347291999996, 42.34410348800003),
    (-71.05378338899999, 42.35742503700004),
    (-71.10056949699998, 42.34938674000006),
    (-71.07889506799995, 42.35324375000005),
    (-71.08023887299998, 42.34534430300005),
    goal
]
print(f"Original stops order: {stops}")

# 2.5) Sort intermediate stops to minimize total network distance between consecutive stops
intermediate = stops[1:-1]
best_order = None
best_dist = float('inf')
# Try all permutations (n! manageable for few stops)
for perm in itertools.permutations(intermediate):
    seq = [start] + list(perm) + [goal]
    # compute total A* path length between seq nodes
    dist = 0
    for a, b in zip(seq, seq[1:]):
        dist += env.astar_path_length(a, b)
    if dist < best_dist:
        best_dist = dist
        best_order = seq
stops = best_order
print(f"Reordered stops for minimal distance (≈ {best_dist * 100} units):")
print(stops)

# 3) Build full route by chaining A* segments between stops
full_path = []
for i in range(len(stops) - 1):
    seg_start = stops[i]
    seg_goal  = stops[i + 1]
    segment = env.astar_path(seg_start, seg_goal)
    # avoid duplication of connecting node
    if i > 0:
        segment = segment[1:]
    full_path.extend(segment)

# 4) Render base map with traffic weights and overlay the full path
plt.ion()
fig, ax = env.render(path=full_path)
plt.show(block=False)

# 5) Overlay stops as red dots
xs, ys = zip(*stops)
ax.scatter(xs, ys, c='red', s=50, marker='o', zorder=5, label='Stops')
ax.legend(loc='upper right')

# 6) Animate the agent moving along the full path
agent_dot, = ax.plot([], [], 'ro', markersize=8)
for node in full_path:
    x, y = node
    agent_dot.set_data([x], [y])
    fig.canvas.draw()
    plt.pause(0.3)

# 7) Finalize display
plt.ioff()
plt.show()
env.close()
