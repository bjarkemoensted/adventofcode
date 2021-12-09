import numpy as np

with open("input09.txt") as f:
    M = np.array([[int(s) for s in line.strip()] for line in f.readlines()])


def get_neighborhood_coords(m, i, j):
    """Takes a matrix (2d np array) and returns a list of coordinates of neighbors.
    Returns 4 points (east, west, north, south), unless input is at the edge/corner."""
    res = []
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    nrows, ncols = m.shape
    for ioff, joff in offsets:
        a = i + ioff
        b = j + joff
        if (0 <= a < nrows) and (0 <= b < ncols):
            res.append((a, b))
        #
    return res


def get_neighborhood_values(m, i, j):
    """Returns a list of the matrix elements around the input point"""
    return [m[a, b] for a, b in get_neighborhood_coords(m, i, j)]


def is_low_point(m, i, j):
    """Determine whether i,j is a 'low point', i.e. m[i,j] has a lower value than all its neighbors."""
    neighbors = get_neighborhood_values(m, i, j)
    val = m[i, j]
    return all(val < neighbor for neighbor in neighbors)


def iterate_coords(m):
    """Iterates over all coordinates i, j for input matrix"""
    nrows, ncols = m.shape
    for i in range(nrows):
        for j in range(ncols):
            yield i, j


# Find all the low points
low_points = []
for i, j in iterate_coords(M):
    if is_low_point(M, i, j):
        low_points.append((i, j))
    #

# Find the sum of risk scores (1 + height of each low points)
risk_scores = [M[i, j] + 1 for i, j in low_points]
star1 = sum(risk_scores)
print(f"Solution to star 2: {star1}.")


def find_unassigned_points(m, unassigned_coords, target_value):
    """Searchers the matrix for a target value, e.g. 0, which is in the set of unassigned crds"""
    return [coords for coords in unassigned_coords if m[coords] == target_value]


def find_bassin_from_low_point(m, i, j):
    """Starts from the lowest point in a bassin. Repeatedly adds surrounding higher points,
    until no higher neighbors with height < 9 exist"""
    bassin = set([])
    new_addition = {(i, j)}
    # As long as we've just added new points, check their neighbors for even higher points
    while new_addition:
        bassin.update(new_addition)
        new_seeds = {tup for tup in new_addition}
        new_addition = set([])
        # Check the neighborhood of newly added points
        for a, b in new_seeds:
            seed_val = m[a, b]
            # If any neighbor is higher up and not already in bassin, add it
            for neighbor in get_neighborhood_coords(m, a, b):
                i_n, j_n = neighbor
                neighbor_val = m[i_n, j_n]
                neighbor_can_flow_down = seed_val < neighbor_val < 9
                if neighbor_can_flow_down and neighbor not in bassin:
                    new_addition.add(neighbor)
                #
            #
        #
    return bassin


# Coordinates that have not yet been assigned to a bassin
unassigned_coords = {(i, j) for i, j in iterate_coords(M) if M[i, j] < 9}

# Result dict - maps each bassin low point to the coordinates contained in the bassin
low_point2basin = {}

# Iterate over the heights of possible bassin low points (0 through 8)
for target_value in range(9):
    low_points = find_unassigned_points(M, unassigned_coords, target_value)
    # For each low point, find the corresponding bassin
    for low_point in low_points:
        i, j = low_point
        bassin = find_bassin_from_low_point(M, i, j)
        # Add to results dict, and remove bassin from the pool of coordinates not assigned to a bassin
        low_point2basin[low_point] = bassin
        unassigned_coords -= bassin
    #


# Find the product of the sizes of the three largest bassins
bassins_by_size = sorted([len(bassin) for bassin in low_point2basin.values()], reverse=True)
star2 = 1
for n in bassins_by_size[:3]:
    star2 *= n

print(f"Solution to star 2: {star2}.")
