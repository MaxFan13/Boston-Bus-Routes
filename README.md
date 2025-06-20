# Optomizing Boston Bus Routes
A python tool that finds the optimal bus stop placements in boston comparing A* and local search with simulated annealing.

## Background
Public transit in dense urban areas often faces challenges with delays and traffic that cause the people who rely on transportation to feel frustrated. The efficiency of public transportation can be improved by optimizing routes and moving stops to encourage people to switch from cars to buses and ultimately reduce traffic. This project uses public Boston traffic data and implements an A* and a local search algorithm to compare the fastest path with the path with most traffic reduction. The result gives a framework for designing more reliable and efficient bus services for the city of Boston.

## Overview:
Astar.py: Original test on base map (no traffic weights)  
a_star.py: a star search code to find the shortest path and the total path cost
boston_area_with_traffic.csv: csv file containing boston traffic data  
boston_street_segmets_sam_system.csv: csv file containing boston map data  
env.py: the map environment  
local_search.py: runs a local search algorithm on the environment  
local_search_helpers.py: local search help functions  
traffic_data_Astar.py: runs an A* search on the environment  
viz.py: visualizations of two models' route comparison, local search performance, and the comparison of two models' performance over multiple rounds
result_viz folder: contains four visualizations generated from viz.py

### How to run:
Run local_search.py or traffic_data_Astar.py to see the respective search algorithm run. Run viz.py to generate visualizations to evalute the performnace of two models. 

## Optimizations:
In our testing, we used the following hyperparameters for local search:  
num_stops=5,  
num_iterations=500,  
initial_temp=0.95,  
cooling_rate=0.98

## Results:
We found A* and local search on average visit the same amount of nodes (160), but A* travels a shorter distance of 12 units while local search averages 16 units. However, A* reduces traffic by -2 while local search has a reduction of -2.5. Overall, our findings align with our expectations: A* finds the shortest route but has less traffic reduction and local search has a longer distance but reduces more traffic. However, our testing was mainly done with the hyperparameters specified above. Tuning the values may drastically change our results. (Further analysis on results can be found on the project report.)

## Future Work:
While our results can be used for future optimization, it doesn’t guarantee that citizens would switch from personal vehicles to public transportation. Transportation routes should also be fixed. Our program doesn’t guarantee the optimized route would positively impact the congestion alleviation all the time. Future work could include implementing real time or more updated traffic data, using other agents (cars) to simulate a more realistic scenario, or using housing and building data to optimize routes matching a majority of passengers’ lifestyle.


## Demo:
A*: https://youtu.be/bmu1eA3km90  
Local Search: https://youtu.be/HkrQokFwQaU

## References:
[1] Zarrinmehr, A., Saffarzadeh, M., & Seyedabrishami, S. (2016). A local search algorithm for finding optimal transit routes configuration with elastic demand. International Transactions in Operational Research, 25(5), 1491–1514. https://doi.org/10.1111/itor.12359

[2] Thi, N. T., Vi, N. T. L., Phung, P. T. K., & Van Can, N. (2023). A simulation - Based optimization for bus route design problem. AIP Conference Proceedings, 2562, 090019. https://doi.org/10.1063/5.0110586

[3] Patel, A. J. (2014). Introduction to the A* algorithm.https://www.redblobgames.com/pathfinding/a-star/introduction.html

[4] heapq — Heap queue algorithm. (n.d.). Python Documentation.https://docs.python.org/3/library/heapq.html


