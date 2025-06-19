import heapq as HQ

def a_star(graph, start, goal, heuristics):
    """
    Find the path using A* search in the given graph

    Uses heapq library as it is efficient and fast. 
    # https://www.geeksforgeeks.org/python/heap-queue-or-heapq-in-python/ 
    # https://docs.python.org/3/library/heapq.html
    # https://www.redblobgames.com/pathfinding/a-star/introduction.html <- came_from

    # Note: our first approach for A star was storing full path in each node, 
    but that design added computation time. Therefore, we optimized it by 
    storing g_score and current node in the queue and reconstructing the path 
    at the end using came_from dictionary that tracks the previous node for each visited node.
    """
    q = []

    # initialize queue
    # (g(n) = c(n) + h(n), current node)
    HQ.heappush(q, ((0+heuristics(start, goal)), start))
    came_from = dict()
    cost_dict = {start:0}

    while len(q)>0:
        g, current = HQ.heappop(q)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], cost_dict[goal]
        for neighbor in graph.neighbors(current):
            edge = graph.get_edge_data(current, neighbor)
            distance = edge.get("length", 1)
            new_c = cost_dict[current] + distance
            if neighbor not in cost_dict or new_c < cost_dict[neighbor]:
                came_from[neighbor] = current
                cost_dict[neighbor] = new_c
                g_score = new_c + heuristics(neighbor, goal)
                HQ.heappush(q, (g_score, neighbor))
