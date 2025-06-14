import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from env import BostonTrafficEnv
import random

# 1) Initialize environment (loads CSVs, builds graph)
env = BostonTrafficEnv(
    street_csv="boston_street_segments_sam_system.csv",
    traffic_csv="boston_area_with_traffic.csv"
)

# 2) Sample a connected start and goal
def sample_start_goal(G):
    comps = list(__import__('networkx').connected_components(G))
    comp = random.choice(comps)
    return random.sample(list(comp), 2)

start, goal = env.sample_start_goal()
print(f"Start: {start}")
print(f"Goal:  {goal}")

# 3) Compute A* path and length
path = env.astar_path(start, goal)
total_length = env.astar_path_length(start, goal)
print(f"Found path with {len(path)} hops, total length ≈ {total_length:.1f} units.")

# 4) Render base map with traffic weights
plt.ion()
fig, ax = env.render(path=path)
plt.show(block=False)

# 5) Animate the agent moving along the path
for node in path:
    x, y = node
    env.agent_dot.set_data([x], [y])
    fig.canvas.draw()
    plt.pause(0.3)

# 6) Finalize
plt.ioff()
plt.show()
env.close()
