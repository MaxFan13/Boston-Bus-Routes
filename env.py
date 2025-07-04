import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely import wkt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from a_star import *

def euclidean(u, v):
    """
    Euclidean distance heuristic for A* on coordinate tuples.
    """
    return np.hypot(u[0] - v[0], u[1] - v[1])


def merge_traffic_weights(original_df: pd.DataFrame, traffic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge traffic weights into the street segment DataFrame.
    Unmatched segments get weight 0.
    """
    merged = original_df.merge(
        traffic_df[['traffic_weight']],
        left_index=True, right_index=True,
        how='left'
    )
    merged['traffic_weight'] = merged['traffic_weight'].fillna(0)
    return merged


class BostonTrafficEnv:
    """
    Environment for Boston street network with traffic weights.

    Methods:
      - sample_start_goal(): pick two connected nodes
      - reset(start=None, goal=None): compute and return full path
      - step(): advance to next node
      - render(path=None): draw map + nodes, edges, colorbar, optional path, and agent dot
      - close(): close figure
      - astar_path: compute A* path
      - astar_path_length: compute path length
    """
    def __init__(
        self,
        street_csv: str = "boston_street_segments_sam_system.csv",
        traffic_csv: str = "boston_area_with_traffic.csv"
    ):
        df = pd.read_csv(street_csv)
        df = df[df["shape_wkt"].notnull() & df["shape_wkt"].apply(lambda x: isinstance(x, str))]
        if 'NBHD_R' in df.columns:
            df = df[df['NBHD_R'] == 'BOSTON']

        traffic_df = pd.read_csv(traffic_csv)
        merged = merge_traffic_weights(df, traffic_df)

        merged['geometry'] = merged['shape_wkt'].apply(wkt.loads)
        self.gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs="EPSG:4326")

        self.G = nx.Graph()
        for _, row in self.gdf.iterrows():
            geom = row.geometry
            parts = [geom] if geom.geom_type == 'LineString' else geom.geoms
            for part in parts:
                u = part.coords[0]
                v = part.coords[-1]
                self.G.add_edge(u, v,
                   length=part.length,
                   traffic_weight=row['traffic_weight'],
                    SEGMENT_ID=row['SEGMENT_ID'])

        self.nodes = list(self.G.nodes)
        self.fig = None
        self.ax = None
        self.agent_dot = None

    def sample_start_goal(self):
        """
        Randomly pick two distinct nodes in the same connected component.
        """
        components = list(nx.connected_components(self.G))
        largest_comp = max(components, key=len)
        start, goal = random.sample(list(largest_comp), 2)
        return start, goal

    def astar_path(self, start, goal):
        """
        Compute A* shortest path from start to goal by 'length'.
        Returns list of node tuples.
        """
        path, cost = a_star(self.G, start, goal, heuristics=euclidean)
        return path
        # return nx.astar_path(
        #     self.G, start, goal,
        #     heuristic=euclidean,
        #     weight='length'
        # )

    def astar_path_length(self, start, goal) -> float:
        """
        Compute total length of A* path.
        """
        path, cost = a_star(self.G, start, goal, heuristics=euclidean)
        return cost
        # return nx.astar_path_length(
        #     self.G, start, goal,
        #     heuristic=euclidean,
        #     weight='length'
        # )

    def reset(self, start=None, goal=None):
        """
        Initialize a new path. If start/goal None, pick a random reachable pair.
        Returns the full path list.
        """
        if start is None or goal is None:
            start = self.nodes[random.randint(0, len(self.nodes) - 1)]
            goal = self.nodes[random.randint(0, len(self.nodes) - 1)]
        self.start = start
        self.goal = goal
        try:
            self.path = self.astar_path(start, goal)
        except nx.NetworkXNoPath:
            raise ValueError(f"No path between {start} and {goal}")
        self._step = 0
        return self.path

    def step(self):
        """
        Advance the agent to the next node on the current path.
        Returns: (node, done)
        """
        if self._step < len(self.path) - 1:
            self._step += 1
        node = self.path[self._step]
        done = (self._step == len(self.path) - 1)
        return node, done

    def render(self, path=None, cmap=plt.cm.RdYlGn_r, figsize=(15,12)):
        """
        Draw the street network with traffic-color edges, node points, colorbar,
        optionally overlay the A* path in blue, and initialize the agent dot.
        Call once before stepping.

        Args:
          path (list of node tuples, optional): sequence to highlight
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        # Position dict
        pos = {n: n for n in self.G.nodes}
        nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            node_size=5,
            node_color='lightblue',
            ax=self.ax
        )
        edge_colors = [d['traffic_weight'] for _, _, d in self.G.edges(data=True)]
        nx.draw_networkx_edges(
            self.G,
            pos=pos,
            edge_color=edge_colors,
            edge_cmap=cmap,
            width=2,
            ax=self.ax
        )
        #Scale for weight
        norm = mcolors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax, fraction=0.03, pad=0.04)
        cbar.set_label('Traffic Weight')
        # Overlay path if provided
        if path and len(path) > 1:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(
                self.G,
                pos=pos,
                edgelist=path_edges,
                edge_color='blue',
                width=2.5,
                ax=self.ax
            )
        # Agent marker
        self.agent_dot, = self.ax.plot([], [], 'ro', markersize=8)
        self.ax.set_title('Boston Traffic Environment')
        return self.fig, self.ax

    def close(self):
        """
        Close the rendering window.
        """
        if self.fig:
            plt.close(self.fig)

# Example usage:
# env = BostonTrafficEnv()
# path = env.reset()
# fig, ax = env.render(path=path)
# for node in path:
#     x, y = node
#     env.agent_dot.set_data([x], [y])
#     fig.canvas.draw()
#     plt.pause(0.3)
# env.close()

# Example usage (use of first two helper functions)
# from local_search_helpers import *
# env = BostonTrafficEnv()
# s, g = env.sample_start_goal() 
# stops = stop_placement(env, s, g, 4)
# routes = generate_routes(stops, env.G)
# route_nodes = [item for item in routes if isinstance(item, tuple)]
# fig, ax = env.render(path=route_nodes, figsize=(12, 8))
# ax.plot(s[0], s[1], marker='o', color='green', markersize=10, label='Start')
# ax.plot(g[0], g[1], marker='*', color='red', markersize=12, label='Goal')
# for i in range(len(stops)):
#     stop = stops[i]
#     if stop != s and stop != g:
#         ax.plot(stop[0], stop[1], marker='s', color='orange', markersize=8, label=f'Stop {i}')
#         ax.text(stop[0], stop[1], str(i), fontsize=9, ha='center', va='center', color='black')
# ax.legend()
# for node in route_nodes:
#     x, y = node
#     env.agent_dot.set_data([x], [y])
#     fig.canvas.draw()
#     plt.pause(0.00001)
# env.close()
