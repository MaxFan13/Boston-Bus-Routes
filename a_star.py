import heapq as HQ

def a_star(graph, start, goal, heuristics):
    """
    Find the path using A* search in the given graph

    Uses heapq library as it is efficient and fast. 
    # https://www.geeksforgeeks.org/python/heap-queue-or-heapq-in-python/ 
    # https://docs.python.org/3/library/heapq.html
    """
    q = []
    visited = set()

    # initialize queue
    # (g(n) = c(n) + h(n), c(n) = cost, current node, path so far)
    HQ.heappush(q, ((0+heuristics(start, goal)), 0, start, [start]))
    while len(q)>0:
        g, c, current, path = HQ.heappop(q)
        if current in visited:
            continue
        if current == goal:
            return path, c
        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
            edge = graph.get_edge_data(current, neighbor)
            distance = edge.get("length", 1)
            new_c = c + distance
            HQ.heappush(q, (new_c + heuristics(neighbor, goal), new_c, neighbor, path + [neighbor]))