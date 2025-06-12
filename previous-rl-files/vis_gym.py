#!/usr/bin/env python3
"""
vis_gym.py

Visualization routines for BostonBusRoutingEnv.  

  • If `vis_gym` (the GeoEnv library) is installed, we produce a live map showing a red dot
    as the bus agent moves from cell to cell.  
  • Otherwise we fall back to a static Matplotlib plot of start/goal.

This module exports:
  - visualize_random_rollout(num_steps=50)
  - visualize_trained_policy(model, num_steps=50)

Both functions import `BostonBusRoutingEnv` from mdp_gym.py and use `fishnet_clipped` and `G`
from boston_map_grid.py.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import bus environment and grid
from mdp_gym import BostonBusRoutingEnv
from boston_map_grid import fishnet_clipped, G

# Attempt to import vis_gym.GeoEnv at runtime
try:
    import vis_gym
    from vis_gym.utils import plot_on_gdf
    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False


def visualize_random_rollout(num_steps: int = 50):
    """
    Creates a new BostonBusRoutingEnv, runs a random policy for up to `num_steps`, and visualizes:
      - If vis_gym is installed: a live map with the agent as a red dot.  
      - Otherwise: a static Matplotlib plot showing start (green) and goal (red).
    """
    env = BostonBusRoutingEnv()
    obs = env.reset()
    start_cell = env.start_cell
    goal_cell = env.goal_cell

    if VIS_AVAILABLE:
        # Prepare a GeoDataFrame with an "idx" column for vis_gym
        vis_gdf = fishnet_clipped.copy().reset_index(drop=True)
        vis_gdf["idx"] = vis_gdf.index

        vg = vis_gym.GeoEnv(gdf=vis_gdf, G=G, start_idx=start_cell, goal_idx=goal_cell)
        vg.reset()

        for t in range(num_steps):
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            vg.step(new_state)
            vg.render()
            if done:
                print("→ Reached Goal!")
                break
        else:
            print("→ Did not reach goal in", num_steps, "steps.")
    else:
        # Static Matplotlib: draw start/goal on the grid
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fishnet_clipped.boundary.plot(ax=ax, color="gray", linewidth=0.3)
        centroids = fishnet_clipped.centroid
        ax.scatter(centroids.x, centroids.y, s=2, color="lightblue", label="Grid Cells")

        start_pt = centroids.iloc[start_cell]
        goal_pt = centroids.iloc[goal_cell]
        ax.scatter(start_pt.x, start_pt.y, s=120, color="green", label="Start")
        ax.scatter(goal_pt.x, goal_pt.y, s=120, color="red",   label="Goal")

        ax.set_title("Boston Grid: Start (green) → Goal (red) [Random Policy]")
        ax.set_axis_off()
        plt.legend()
        plt.show()

    print(f"Visited {len(env.visited)} unique cells out of {env.n_cells}.")


def visualize_trained_policy(model, num_steps: int = 50):
    """
    Given a trained RL model (e.g. from Stable Baselines3), runs one rollout and visualizes
    the agent following its learned policy. Behavior is the same as visualize_random_rollout,
    except that actions are chosen by `model.predict(obs)`.
    """
    env = BostonBusRoutingEnv()
    obs = env.reset()
    start_cell = env.start_cell
    goal_cell = env.goal_cell

    if VIS_AVAILABLE:
        vis_gdf = fishnet_clipped.copy().reset_index(drop=True)
        vis_gdf["idx"] = vis_gdf.index

        vg = vis_gym.GeoEnv(gdf=vis_gdf, G=G, start_idx=start_cell, goal_idx=goal_cell)
        vg.reset()

        for t in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            new_state, reward, done, info = env.step(int(action))
            vg.step(new_state)
            vg.render()
            if done:
                print("→ Reached Goal!")
                break
        else:
            print("→ Did not reach goal in", num_steps, "steps.")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fishnet_clipped.boundary.plot(ax=ax, color="gray", linewidth=0.3)
        centroids = fishnet_clipped.centroid
        ax.scatter(centroids.x, centroids.y, s=2, color="lightblue", label="Grid Cells")

        start_pt = centroids.iloc[start_cell]
        goal_pt = centroids.iloc[goal_cell]
        ax.scatter(start_pt.x, start_pt.y, s=120, color="green", label="Start")
        ax.scatter(goal_pt.x, goal_pt.y, s=120, color="red",   label="Goal")

        ax.set_title("Boston Grid: Start (green) → Goal (red) [Trained Policy]")
        ax.set_axis_off()
        plt.legend()
        plt.show()

    print(f"Visited {len(env.visited)} unique cells out of {env.n_cells}.")
