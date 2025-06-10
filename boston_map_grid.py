import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point
import networkx as nx
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("boston_street_segments_sam_system.csv")

# Filter out rows with invalid or missing WKT geometry strings
df = df[df["shape_wkt"].notnull() & df["shape_wkt"].apply(lambda x: isinstance(x, str))]

# Convert WKT strings to shapely geometry objects
df["geometry"] = df["shape_wkt"].apply(wkt.loads)

# Create GeoDataFrame with correct CRS (WGS84)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

print("Total rows:", len(df))
print("Rows with valid geometry:", len(gdf))

# Reproject to NAD83 / Massachusetts Mainland for accurate plotting and measurements
gdf = gdf.to_crs("EPSG:26986")

gdf['NBHD_R'].unique()

# Step 1: Filter GeoDataFrame to only 'BOSTON'
boston_gdf = gdf[gdf['NBHD_R'] == 'BOSTON']

# Step 2: Build a new graph from BOSTON data only
G_boston = nx.Graph()

for idx, row in boston_gdf.iterrows():
    geom = row.geometry
    if isinstance(geom, LineString):
        geoms = [geom]
    elif isinstance(geom, MultiLineString):
        geoms = geom.geoms
    else:
        continue

    for part in geoms:
        start = part.coords[0]
        end = part.coords[-1]
        G_boston.add_edge(start, end, length=part.length, segment_id=row['SEGMENT_ID'])

# Print total number of nodes and edges
print("Total nodes:", G_boston.number_of_nodes())
print("Total edges:", G_boston.number_of_edges())

# --- Pick a node (e.g., first one) ---
example_node = list(G_boston.nodes)[0]
print("\nExample node:", example_node)

# --- Neighbors of the node ---
neighbors = list(G_boston.neighbors(example_node))
print("Neighbors:", neighbors)

# --- Edges connected to the node and their attributes (weights, etc.) ---
for neighbor in neighbors:
    edge_data = G_boston.get_edge_data(example_node, neighbor)
    print(f"Edge from {example_node} to {neighbor}:")
    print(f"  Length: {edge_data['length']:.2f}")
    print(f"  Segment ID: {edge_data['segment_id']}")

# --- All edges and their weights (lengths) ---
print("\nAll edges and their weights (first 5 shown):")
for u, v, data in list(G_boston.edges(data=True))[:5]:
    print(f"Edge: {u} <-> {v}, Length: {data['length']:.2f}")


import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiLineString
import geopandas as gpd

# --- ASSUMES your G_boston and boston_gdf are already in memory ---
# If not, re-run your snippet to build G_boston and boston_gdf first:
#
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString
import random
"""
gdf = gpd.read_file("your_street_segments.geojson")
boston_gdf = gdf[gdf['NBHD_R'] == 'BOSTON']
G_boston = nx.Graph()
for idx, row in boston_gdf.iterrows():
    geom = row.geometry
    geoms = [geom] if isinstance(geom, LineString) else geom.geoms
    for part in geoms:
        start = part.coords[0]
        end   = part.coords[-1]
        G_boston.add_edge(start, end, length=part.length, segment_id=row['SEGMENT_ID'])
"""
# For this demo, let’s assume G_boston is already defined:

# 1) Pick two example nodes (you can replace with any nodes in G_boston)
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
pos = {n: n for n in G_boston.nodes}

# 5) Plot the entire graph in light grey
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
nx.draw(G_boston, pos=pos, node_size=5, edge_color='lightgray', ax=ax)

# 6) Overlay the A* path in blue
path_edges = list(zip(path[:-1], path[1:]))
nx.draw_networkx_edges(G_boston, pos=pos, edgelist=path_edges,
                       edge_color='blue', width=2.0, ax=ax)

# 7) Animate the agent as a red dot
agent_dot, = ax.plot([], [], 'ro', markersize=8)
for node in path:
    x, y = node
    agent_dot.set_data([x], [y])
    fig.canvas.draw()
    plt.pause(0.3)

plt.ioff()
plt.show()
