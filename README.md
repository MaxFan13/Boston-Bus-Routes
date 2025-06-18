# Optomizing Boston Bus Routes
A python tool that finds the optimal bus stop placements in boston comparing A* and local search with simulated annealing.

## Background
Public transit in dense urban areas often faces challenges with delays and traffic that cause the people who rely on transportation to feel frustrated. The efficiency of public transportation can be improved by optomizing routes and moving stops to enncourage people to switch from cars to buses and ultimately reduce traffic. This project uses public Boston traffic data and implements an A* and local search algorithm to compare the fastest path with the path with most traffic redduction. The result gives a framework for designing more reliable and effficient bus services for the city of Boston.

## How it's made:
We used two datasets, one from google maps and another straight from the Boston government (https://data.boston.gov/dataset/traffic-related-data), that detail the street segments, traffic, coordinates, and more of the city of Boston. We created a environment map of Boston using GeoPandas and overlayed the traffic weights using networkX. Then we built the A* and local search algorithms. For the local search, we developed 3 helper functions:
1) stop_placement(): randomly generates stops on the route
2) generate_routes(): stitches stops via the shortest path
3) score_calculator(): returns a score based on the traffic reduced
We made this a minimization problem where we tried getting the reward as low (negative) as possible. Our penalites include any increase in the path length and visting more nodes.

## Optimizations:
In our testing, we used the following hyperparameters for local search:
num_stops=5,
num_iterations=500,
initial_temp=0.95,
cooling_rate=0.98

## Results:
We found A* and local search on average visit the same amount of nodes (160), but A* travels a shorter distance of 12 units while local search averages 16 units. However, A* reduces traffic by -2 while local search has a reduction of -2.5. Overall, our findings align with our expectations: A* finds the shortest route but has less traffic reduction and local search has a longer distance but reduces more traffic. However, our testing was mainly done with the hyperparameters specified above. Tuning the values may drastically change our results.

## Future Work:
While our results can be used for future optimization, it doesn’t guarantee that citizens would switch from personal vehicles to public transportation. Transportation routes should also be fixed. Our program doesn’t guarantee the optimized route would positively impact the congestion alleviation all the time. Future work could include implementing real time or more updated traffic data, using other agents (cars) to simulate a more realistic scenario, or using housing and building data to optimize routes matching a majority of passengers’ lifestyle.




## Demo:
A*: https://youtu.be/bmu1eA3km90
Local Search: https://youtu.be/HkrQokFwQaU
