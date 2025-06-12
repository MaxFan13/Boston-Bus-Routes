#!/usr/bin/env python3
"""
mdp_gym.py

Defines the BostonBusRoutingEnv MDP:
  - State: a single grid‐cell index (0 … n_cells-1)
  - Action: Discrete(4) → {0=N,1=E,2=S,3=W}
  - Transition: Moves to neighbor in that direction if valid, else stays
  - Reward: –1.0 per move (uniform cost), +100 if reaching goal
  - Episode ends when agent hits goal cell

It imports the clipped grid (fishnet_clipped), adjacency graph (G), and served_cells mask
from boston_map_grid.py (which must be present in the same directory).
"""

import numpy as np
import gym
from gym import spaces

# Import precomputed grid + adjacency from boston_map_grid.py
from boston_map_grid import fishnet_clipped, G, served_cells

# Cell size constant (matches boston_map_grid.py)
CELL_SIZE = 500  # meters per cell


class BostonBusRoutingEnv(gym.Env):
    """
    Gym environment for a single bus agent on the Boston grid.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, start_cell: int = None, goal_cell: int = None):
        super(BostonBusRoutingEnv, self).__init__()

        # Load the clipped grid and adjacency graph
        self.grid_gdf = fishnet_clipped.reset_index(drop=True)
        self.adj_graph = G
        self.n_cells = len(self.grid_gdf)

        # Discrete(4) actions: 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST
        self.action_space = spaces.Discrete(4)
        # Observation: current cell index ∈ [0, n_cells-1]
        self.observation_space = spaces.Discrete(self.n_cells)

        # If start_cell not provided, pick a random "served" cell (or any if none served)
        if start_cell is None:
            if np.count_nonzero(served_cells) > 0:
                starts = np.where(served_cells == 1)[0]
                self.start_cell = int(np.random.choice(starts))
            else:
                self.start_cell = int(np.random.choice(self.n_cells))
        else:
            self.start_cell = start_cell

        # If goal_cell not provided, pick a different random served cell (or any other)
        if goal_cell is None:
            if np.count_nonzero(served_cells) > 1:
                served_idxs = np.where(served_cells == 1)[0].tolist()
                if self.start_cell in served_idxs:
                    served_idxs.remove(self.start_cell)
                if served_idxs:
                    self.goal_cell = int(np.random.choice(served_idxs))
                else:
                    all_cells = list(range(self.n_cells))
                    all_cells.remove(self.start_cell)
                    self.goal_cell = int(np.random.choice(all_cells))
            else:
                all_cells = list(range(self.n_cells))
                all_cells.remove(self.start_cell)
                self.goal_cell = int(np.random.choice(all_cells))
        else:
            self.goal_cell = goal_cell

        self.state = self.start_cell
        self.visited = set([self.start_cell])

    def reset(self):
        """Resets to start_cell and clears visited set."""
        self.visited = set([self.start_cell])
        self.state = self.start_cell
        return self.state

    def step(self, action: int):
        """
        Takes action ∈ {0=N,1=E,2=S,3=W}. Moves to neighbor if valid, else stays in place.
        Returns (new_state, reward, done, info):
          - reward = –1.0 + (100 if new_state == goal_cell)
          - done = True if new_state == goal_cell
          - info = dict with travel_time and visited_count
        """
        current = self.state
        cur_centroid = self.grid_gdf.geometry.iloc[current].centroid
        neighbors = list(self.adj_graph.neighbors(current))

        new_state = current
        best_dist = float("inf")

        # Find the neighbor whose centroid matches the chosen compass direction
        for n in neighbors:
            n_centroid = self.grid_gdf.geometry.iloc[n].centroid
            dx = n_centroid.x - cur_centroid.x
            dy = n_centroid.y - cur_centroid.y

            if action == 0 and (dy > 0 and abs(dx) < CELL_SIZE / 2):   # North
                dist = abs(dy)
            elif action == 1 and (dx > 0 and abs(dy) < CELL_SIZE / 2): # East
                dist = abs(dx)
            elif action == 2 and (dy < 0 and abs(dx) < CELL_SIZE / 2): # South
                dist = abs(dy)
            elif action == 3 and (dx < 0 and abs(dy) < CELL_SIZE / 2): # West
                dist = abs(dx)
            else:
                continue

            if dist < best_dist:
                best_dist = dist
                new_state = n

        # Uniform cost = 1.0 minute (or staying in place)
        travel_time = 1.0

        done = (new_state == self.goal_cell)
        reward = -travel_time + (100.0 if done else 0.0)

        self.state = new_state
        self.visited.add(self.state)

        info = {
            "travel_time": travel_time,
            "visited_count": len(self.visited)
        }
        return self.state, reward, done, info

    def render(self, mode="human"):
        """Textual render: current cell, goal, visited so far."""
        print(f"Agent at cell {self.state} | Goal = {self.goal_cell} | Visited = {len(self.visited)}")

    def seed(self, seed=None):
        np.random.seed(seed)
