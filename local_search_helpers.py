"""
local_search_helpers.py

Contains helper functions for Simmulated Annealing Local Search

  - stop_placement(start, goal, num_stops) -> list of stops (including start & goal)
  - generate_routes(stops) -> list of nodes/edges representing full movement sequence
  - score_calculator(route) -> numeric score (lower is better)

"""

import networkx as nx
import random

def stop_placement(env, start, goal, num_stops):
    components = list(nx.connected_components(env.G))
    largest_comp = max(components, key=len)
    possible_stops = [node for node in largest_comp if node not in [start, goal]]
    stops = random.sample(possible_stops, num_stops)
    stop_sequence = [start] + stops + [goal]

    return stop_sequence

def generate_routes(stop_sequence, graph):
    resulting_route = []

    for i in range(len(stop_sequence) - 1):
        from_point = stop_sequence[i]
        to_point = stop_sequence[i+1]

        ## Get a path between two points from graph 
        from_to_path = nx.shortest_path(graph, from_point, to_point, 'weight')

        for j in range(len(from_to_path) - 1):
            from_stop, to_stop = from_to_path[j], from_to_path[j + 1]
            edge = graph.get_edge_data(from_stop, to_stop)

            if len(resulting_route) == 0:
                resulting_route.append(from_stop)

            # Use the segment id to represent the edge - can change if needed
            resulting_route.append(edge['segment_id'])
            
            resulting_route.append(to_stop)

    return resulting_route
