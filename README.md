# Optomizing Boston Bus Routes
A python tool that finds the optimal bus stop placements in boston comparing A* and local search with simulated annealing.

## Background
Public transit in dense urban areas often faces challenges with delays and traffic that cause the people who rely on transportation to feel frustrated. The efficiency of public transportation can be improved by optimizing routes and moving stops to encourage people to switch from cars to buses and ultimately reduce traffic. This project uses public Boston traffic data and implements an A* and a local search algorithm to compare the fastest path with the path with most traffic reduction. The result gives a framework for designing more reliable and efficient bus services for the city of Boston.

## Overview:
Astar.py: Original test on base map (no traffic weights)  
boston_area_with_traffic.csv: csv file containing boston traffic data  
boston_street_segmets_sam_system.csv: csv file containing boston map data  
env.py: the map environment  
local_search.py: runs a local search algorithm on the environment  
local_search_helpers.py: local search help functions  
traffic_data_Astar.py: runs an A* search on the environment  

### How to run:
Run local_search.py or traffic_data_Astar.py to see the respective search algorithm run.

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
