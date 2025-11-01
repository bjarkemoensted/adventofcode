import copy

import numpy as np

with open("input11.txt") as f:
    M = np.array([[int(s) for s in line.strip()] for line in f], dtype=int)


def get_neighborhood_coords(m, i, j):
    """Takes a matrix (2d np array) and returns a list of coordinates of neighbors.
    Returns 8 points (E, W, S, N, SW etc), unless input is at the edge/corner."""
    res = []
    nrows, ncols = m.shape
    for ioff in range(-1, 2):
        for joff in range(-1, 2):
            if ioff == joff == 0:
                continue
            a = i + ioff
            b = j + joff
            if (0 <= a < nrows) and (0 <= b < ncols):
                res.append((a, b))
            #
        #
    return res


def iterate_coords(m):
    """Iterates over all coordinates i, j for input matrix"""
    nrows, ncols = m.shape
    for i in range(nrows):
        for j in range(ncols):
            yield i, j
        #
    #


def step(m):
    # Increment all energy levels by one
    inc = np.ones(shape=M.shape, dtype=int)
    m += inc

    # We're done if all energy levels are below the flash threshold
    # Find those about to flash
    already_flashed = set([])
    about_to_flash = set([(i, j) for i, j in iterate_coords(m) if m[i, j] >= 10])

    while about_to_flash:
        # Matrix to keep track of which elements to increment
        blast = np.zeros(shape=m.shape, dtype=int)

        for i, j in about_to_flash:
            # Update set of octopi that already flashed
            already_flashed.add((i, j))
            # Increment the light blast in the vicinity of each such octopus
            for ii, jj in get_neighborhood_coords(m, i, j):
                blast[ii, jj] += 1
            #

        # Flash neighbors
        m += blast

        # Find those about to flash again. Done if none.
        remaining_coords = set(iterate_coords(m)) - already_flashed
        about_to_flash = set([(i, j) for i, j in remaining_coords if m[i, j] >= 10])

    # Remove energy from those who flashed
    n_flashed = len(already_flashed)
    for i, j in already_flashed:
        m[i, j] = 0

    return n_flashed


# Operate on a copy of the input
M1 = copy.deepcopy(M)
n_flashes = 0  # Count how many luminescent octopi flash during 100 iterations
for i in range(100):
    n_flashes_this_round = step(M1)
    n_flashes += n_flashes_this_round

print(f"Solution to star 1 is {n_flashes}.")

# Find the first iteration where all octopi flash at the same time
M2 = copy.deepcopy(M)
n = 0
while not all(v == 0 for v in M2.flat):
    step(M2)
    n += 1

print(f"Solution to star 2: {n}.")
