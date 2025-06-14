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

        if not resulting_route:
            resulting_route.extend(from_to_path)
        else:
            resulting_route.extend(from_to_path[1:])

    return resulting_route

def score_calculator(route, graph):
    traversal_counts = {}
    total_traffic_reduction = 0.0
    multipliers = [0.5, 0.25, 0.1, 0.05]
    nearby_multiplier = 0.05
    
    processed_nearby_edges = set()

    for i in range(len(route) - 1):
        # reduce traffic with diminishing returns after a couple of traversals

        u, v = route[i], route[i+1]

        edge = graph.get_edge_data(u, v)

        segment_id = edge.get('SEGMENT_ID')
        traffic = edge.get('traffic_weight', 0)

        count = traversal_counts.get(segment_id, 0)

        if count < len(multipliers):
            multiplier = multipliers[count]
        else:
            multiplier = 0.0
        
        reduction = traffic * multiplier
        total_traffic_reduction += reduction
        traversal_counts[segment_id] = count + 1

        # reduce traffic of nearby edges

        for neighbor_node in list(graph.neighbors(u)) + list(graph.neighbors(v)):
            for start_node, end_node in [(u, neighbor_node), (v, neighbor_node)]:
                if start_node == end_node: continue

                if (start_node == u and end_node == v) or (start_node == v and end_node == u):
                    continue

                nearby_edge_data = graph.get_edge_data(start_node, end_node)

                if nearby_edge_data is not None:
                    edge_identifier = frozenset([start_node, end_node])
                    if edge_identifier not in processed_nearby_edges:
                        nearby_traffic = nearby_edge_data.get('traffic_weight', 0)
                        total_traffic_reduction += nearby_traffic * nearby_multiplier
                        processed_nearby_edges.add(edge_identifier)


    return -total_traffic_reduction
    