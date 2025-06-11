import json
import requests
import pandas as pd
import geopandas as gpd
from shapely import wkt
import time
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt

df = pd.read_csv("boston_street_segments_sam_system.csv")
df = df[df["shape_wkt"].notnull() & df["shape_wkt"].apply(lambda x: isinstance(x, str))]
df = df[df['NBHD_R'] == 'BOSTON']

# Process all simple segments
#result_df = calculate_traffic_weights_optimized(df_major_roads, use_simple_segments_only=True)
result_df = pd.read_csv("boston_area_with_traffic.csv")

def merge_traffic_weights(original_df, traffic_df):
    merged_df = original_df.copy()
    
    if 'traffic_weight' in traffic_df.columns:
        traffic_weights = traffic_df[['traffic_weight']].copy()
        merged_df = merged_df.merge(traffic_weights, left_index=True, right_index=True, how='left')
        merged_df['traffic_weight'] = merged_df['traffic_weight'].fillna(0)
        
        print(f"Merged {len(original_df)} total segments")
        print(f"- {len(traffic_df[traffic_df['traffic_weight'].notnull()])} with traffic weights")
        print(f"- {len(merged_df[merged_df['traffic_weight'] == 0])} assigned weight 0")
    else:
        merged_df['traffic_weight'] = 0
        print("No traffic weights found, all segments assigned weight 0")
    
    return merged_df

df_complete = merge_traffic_weights(df, result_df)

print(f"Final dataset: {len(df_complete)} segments with traffic_weight column")


import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("boston_area_with_traffic.csv")
df_viz = df[df['traffic_weight'].notnull()].copy()

if len(df_viz) == 0:
    print("No traffic weight data to visualize")

df_viz['geometry'] = df_viz['shape_wkt'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df_viz)

fig, ax = plt.subplots(1, 1, figsize=(15, 12))

weights = df_viz['traffic_weight'].values

gdf.plot(ax=ax, 
        column='traffic_weight',
        cmap='RdYlGn_r',
        linewidth=2,
        legend=True,
        vmin=0,
        vmax=1)

ax.set_title('Boston Street Segments - Traffic Weights', fontsize=16)

stats_text = f"Min: {weights.min():.3f}, Max: {weights.max():.3f}, Mean: {weights.mean():.3f}"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top')

plt.show()

#visualize_traffic_weights(df)


import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString
import geopandas as gpd
#df["geometry"] = df["shape_wkt"].apply(wkt.loads)

# Create GeoDataFrame with correct CRS (WGS84)
#gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
#gdf = gdf.to_crs("EPSG:26986")
G_boston = nx.Graph()

for idx, row in gdf.iterrows():
    geom = row.geometry
    # break MultiLineStrings into parts
    parts = [geom] if geom.geom_type == 'LineString' else geom.geoms

    for part in parts:
        start = part.coords[0]
        end   = part.coords[-1]
        length = part.length
        traffic = row['traffic_weight']

        # attach both length and traffic_weight to each edge
        G_boston.add_edge(start, end,
                   length=length,
                   traffic_weight=traffic)




import random
nodes = list(G_boston.nodes)
start = nodes[random.randint(0, len(nodes) - 1)]
goal  = nodes[random.randint(0, len(nodes) - 1)]
print("Start:", start)
print("Goal: ", goal)

# 2) Define the Euclidean heuristic
def euclidean(u, v):
    ux, uy = u
    vx, vy = v
    return np.hypot(ux - vx, uy - vy)

# 3) Compute the A* shortest path (by 'length' attribute)
path = nx.astar_path(G_boston, start, goal,
                     heuristic=euclidean,
                     weight='length')
total_length = nx.astar_path_length(G_boston, start, goal,
                                    heuristic=euclidean,
                                    weight='length')
print(f"Found path with {len(path)} hops, total length ≈ {total_length:.1f} units.")

# 4) Prepare for animation: positions are the coordinates themselves
import matplotlib.colors as mcolors

# after you build G_boston so that each edge has d['traffic_weight']
pos = {n: n for n in G_boston.nodes()}
plt.ion()
fig, ax = plt.subplots(figsize=(15, 12))

# draw edges, colored by traffic_weight
edge_colors = [d['traffic_weight'] for u, v, d in G_boston.edges(data=True)]

nx.draw(
    G_boston,
    pos=pos,
    node_size=5,
    edge_color=edge_colors,
    edge_cmap=plt.cm.RdYlGn_r,
    width=2,
    ax=ax
)

# add a colorbar matching the range of your data
norm = mcolors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)/2)
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Traffic Weight')

ax.set_title('Boston Street Segments – Traffic Weights')



# 6) Overlay the A* path in blue
path_edges = list(zip(path[:-1], path[1:]))
nx.draw_networkx_edges(G_boston, pos=pos, edgelist=path_edges, edge_color='blue', width=2.0, ax=ax)

# 7) Animate the agent as a red dot
agent_dot, = ax.plot([], [], 'ro', markersize=8)
for node in path:
    x, y = node
    print(node)
    agent_dot.set_data([x], [y])
    fig.canvas.draw()
    plt.pause(0.3)

plt.ioff()
plt.show()