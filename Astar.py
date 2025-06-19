import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from a_star import *

# Import the pre‐computed grid and adjacency graph
from boston_map_grid import fishnet_clipped, G

# 1) Pick two random distinct grid cells as start and goal
start, goal = random.sample(list(G.nodes), 2)
print("Start cell:", start)
print("Goal cell: ", goal)

# 2) Precompute centroids for heuristic
centroids = fishnet_clipped.geometry.centroid

def heuristic(u, v):
    x1, y1 = centroids.iloc[u].coords[0]
    x2, y2 = centroids.iloc[v].coords[0]
    return np.hypot(x1 - x2, y1 - y2)

# 3) Run A* (using the edge attribute “weight” for cost)
path = astar_path(G, start, goal, heuristic=heuristic, weight='weight')
path_length = astar_path_length(G, start, goal, heuristic=heuristic, weight='weight')
print(f"Found path of {len(path)} steps, total distance ≈ {path_length:.2f}")

# 4) Prepare for plotting and animation
pos = {i: centroids.iloc[i].coords[0] for i in G.nodes()}

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
# draw full grid
nx.draw(G, pos=pos, node_size=2, edge_color='lightgray', ax=ax)
# highlight the A* path
nx.draw_networkx_edges(G, pos=pos, edgelist=list(zip(path, path[1:])),
                       edge_color='blue', width=2, ax=ax)

# animate a red dot moving along the path
dot, = ax.plot([], [], 'ro', markersize=6)
for node in path:
    x, y = pos[node]
    dot.set_data([x], [y])
    fig.canvas.draw()
    plt.pause(0.2)

plt.ioff()
plt.show()
