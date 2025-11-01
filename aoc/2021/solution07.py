# Read in data
with open("input07.txt") as f:
    numbers = [int(s) for s in f.read().strip().split(",")]

# List potential target values where the crab submarines can line up
targets = range(min(numbers), max(numbers))


def _default_metric(a, b):
    return abs(a - b)


def dist(arr, target, metric=None):
    """Computes the sum of distances to target, given a distance metric"""
    if not metric:
        metric = _default_metric
    return sum(metric(val, target) for val in arr)


# Find the shortest distance
best_target = min(targets, key=lambda x: dist(numbers, x))
shortest_distance = dist(numbers, best_target)
print(f"Solution to star 1: {shortest_distance}.")


# Traveling a distance of x now costs 1 + 2 + ... + n. Define metric for this.
def dist_quad(a, b):
    d = abs(a - b)
    res = int(d * (d + 1) / 2) # Gauss trick: 1+2+...+n = n(n+1)/2
    return res


# Compute shortest distance using the quadratic metric thingy.
best_target2 = min(targets, key=lambda x: dist(numbers, x, dist_quad))
shortest_distance2 = dist(numbers, best_target2, dist_quad)
print(f"Solution to star 2: {shortest_distance2}.")
