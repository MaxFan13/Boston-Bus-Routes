#!/usr/bin/env python3
"""
Q_learning.py

Implements a simple tabular Q‐learning loop for BostonBusRoutingEnv:

  - Runs for a specified number of episodes.
  - Uses an ε‐greedy policy with decaying ε.
  - Updates Q(s,a) via the standard Bellman update and a learning rate η = 1 / (1 + visits(s,a)).
  - At the end, saves the learned Q‐table to 'boston_Q_table.pkl'.

Usage:
    python Q_learning.py
"""

import pickle
import numpy as np

# Import our environment
from mdp_gym import BostonBusRoutingEnv


def Q_learning(num_episodes=500, gamma=0.9, epsilon=1.0, decay_rate=0.995):
    """
    Run Q‐learning on BostonBusRoutingEnv.

    - num_episodes: how many episodes to run
    - gamma: discount factor
    - epsilon: initial exploration rate
    - decay_rate: multiply epsilon by this after each episode

    Returns:
    - Q_table: dict mapping state_index → np.array of length action_space.n
    """
    env = BostonBusRoutingEnv()
    num_actions = env.action_space.n

    # Q_table: dict[state_index] → np.array([Q(s,0), Q(s,1), ..., Q(s,num_actions-1)])
    Q_table = {}
    # visit_counts[(state,action)] → number of times Q(s,a) was updated
    visit_counts = {}

    for ep in range(num_episodes):
        obs_state = env.reset()
        state = obs_state  # in our MDP, the observation _is_ the state index

        if state not in Q_table:
            Q_table[state] = np.zeros(num_actions, dtype=float)

        done = False
        while not done:
            # ε‐greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = int(np.argmax(Q_table[state]))

            new_state, reward, done, info = env.step(action)

            if new_state not in Q_table:
                Q_table[new_state] = np.zeros(num_actions, dtype=float)

            # Compute learning rate η = 1 / (1 + visits(s,a))
            sa = (state, action)
            count = visit_counts.get(sa, 0)
            eta = 1.0 / (1.0 + count)
            visit_counts[sa] = count + 1

            # Q‐learning update: Q(s,a) ← Q(s,a) + η [r + γ max_a' Q(s',a') – Q(s,a)]
            best_next = np.max(Q_table[new_state])
            td_target = reward + gamma * best_next
            td_error = td_target - Q_table[state][action]
            Q_table[state][action] += eta * td_error

            state = new_state

        # Decay ε but keep it ≥ 0.05
        epsilon = max(0.05, epsilon * decay_rate)

    return Q_table


if __name__ == "__main__":
    # Hyperparameters
    NUM_EPISODES = 1000
    GAMMA = 0.9
    INITIAL_EPS = 1.0
    EPS_DECAY = 0.995

    print("Starting Q‐learning...")
    Q_table = Q_learning(
        num_episodes=NUM_EPISODES,
        gamma=GAMMA,
        epsilon=INITIAL_EPS,
        decay_rate=EPS_DECAY
    )
    print("Q‐learning complete. Saving Q‐table to 'boston_Q_table.pkl'…")

    with open("boston_Q_table.pkl", "wb") as f:
        pickle.dump(Q_table, f)

    print("Done.")
