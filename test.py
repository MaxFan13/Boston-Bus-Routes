from env import BostonTrafficEnv
import matplotlib.pyplot as plt

env = BostonTrafficEnv(
    street_csv="boston_street_segments_sam_system.csv",
    traffic_csv="boston_area_with_traffic.csv"
)

# Option A: auto-pick a start/goal
plt.ion()
path = env.reset()
fig, ax = env.render(path=path)
for node in path:
    x, y = node
    env.agent_dot.set_data([x], [y])
    fig.canvas.draw()
    plt.pause(0.3)


"""
# Option B: specify your own start/goal
start, goal = env.sample_start_goal()
path = env.reset(start, goal)
# … same loop …
"""