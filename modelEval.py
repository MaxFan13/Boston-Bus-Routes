"""
model_eval.py

Evaluates multiple paths of same route and determines best one

"""


# Takes in list of prospective paths for one route, scores each of them, and returns best one
def scorePaths(paths):
    scoredPaths = []
    for path in paths:
        # TBD, weights and features of each part of path total up to score
        continue

    bestScore = max(scoredPaths)

    return (paths[scoredPaths.index(bestScore)])

