from gym import spaces
import numpy as np

CELL_SIZE = 500

class BostonGridEnv(gym.Env):
    """
    A minimal OpenAI Gym environment:
      - State = integer index of a grid cell [0 .. num_cells-1]
      - Action space = Discrete(4): 0=North,1=East,2=South,3=West
      - Agent moves to the neighbor in that direction if available, otherwise stays put
      - Reward = +1 if agent reaches goal cell, else 0
    """

    metadata = {'render.modes': ['human']}
    

    def __init__(self, grid_gdf, adj_graph, start_cell=None, goal_cell=None):
        super(BostonGridEnv, self).__init__()

        self.grid_gdf = grid_gdf.reset_index(drop=True)
        self.adj_graph = adj_graph
        self.num_cells = len(self.grid_gdf)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.num_cells)

        # Choose a random start/goal if not provided
        if start_cell is None:
            self.start_cell = np.random.choice(self.num_cells)
        else:
            self.start_cell = start_cell

        if goal_cell is None:
            choices = list(range(self.num_cells))
            choices.remove(self.start_cell)
            self.goal_cell = np.random.choice(choices)
        else:
            self.goal_cell = goal_cell

        self.state = self.start_cell

    def reset(self):
        self.state = self.start_cell
        return self.state

    def step(self, action):
        current = self.state
        cur_centroid = self.grid_gdf.loc[current].geometry.centroid
        neighbors = list(self.adj_graph.neighbors(current))

        new_state = current
        if neighbors:
            cx, cy = cur_centroid.x, cur_centroid.y
            best_dist = float('inf')
            best_cell = current

            for n in neighbors:
                nc = self.grid_gdf.loc[n].geometry.centroid
                dx, dy = nc.x - cx, nc.y - cy

                if action == 0 and (dy > 0 and abs(dx) < CELL_SIZE/2):  # North
                    dist = abs(dy)
                elif action == 1 and (dx > 0 and abs(dy) < CELL_SIZE/2):  # East
                    dist = abs(dx)
                elif action == 2 and (dy < 0 and abs(dx) < CELL_SIZE/2):  # South
                    dist = abs(dy)
                elif action == 3 and (dx < 0 and abs(dy) < CELL_SIZE/2):  # West
                    dist = abs(dx)
                else:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_cell = n

            new_state = best_cell

        self.state = new_state
        done = (self.state == self.goal_cell)
        reward = 1.0 if done else 0.0
        info = {}
        return self.state, reward, done, info

    def render(self, mode='human'):
        print(f"Agent at cell {self.state}, Goal = {self.goal_cell}.")

    def seed(self, seed=None):
        np.random.seed(seed)