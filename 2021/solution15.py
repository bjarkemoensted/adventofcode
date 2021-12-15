import copy
import networkx as nx
import numpy as np


with open("input15.txt") as f:
    cave = np.array([[int(s) for s in list(line.strip())] for line in f.readlines()])


def get_neighborhood_coords(m, i, j):
    """Takes a matrix (2d np array) and returns a list of coordinates of neighbors.
    Returns 8 points (E, W, S, N, SW etc), unless input is at the edge/corner."""
    res = []
    nrows, ncols = m.shape
    for ioff in range(-1, 2):
        for joff in range(-1, 2):
            if ioff == joff == 0:
                continue
            if abs(ioff) == abs(joff):
                continue  # Don't allow diagonals
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


def build_graph_of_cave(cave):
    """Constructs a directional graph of a cave"""
    G = nx.DiGraph()
    # Connect each node to its neighbors, using neighbor risk as edge weight
    for i, j in iterate_coords(cave):
        u = (i, j)
        for ii, jj in get_neighborhood_coords(cave, i, j):
            v = (ii, jj)
            weight = cave[ii, jj]
            G.add_edge(u, v, weight=weight)
        #
    return G


G = build_graph_of_cave(cave)
enter = (0,0)
exit = tuple(v - 1 for v in cave.shape)
# Find the shortest path through the cave
path = nx.shortest_path(G, enter, exit, weight="weight")
risk = sum([cave[i, j] for i, j in path[1:]])  # Remember the first step doesn't count
print(f"Solution to star 1: {risk}.")


def grow_larger_cave(cave, factor=5, max_level=9):
    """Extends a cave in a 5x5 'grid'.
    On grid i, j, the risk of each cell is incremented by i+j, with
    numbers above 9 wrapping around from one (10 -> 1, 11 -> 2, etc)."""
    nrows, ncols = cave.shape
    a, b = factor*nrows, factor*ncols
    res = np.zeros(shape=(a, b), dtype=int)

    for i in range(factor):
        for j in range(factor):
            growth = i + j
            section = copy.deepcopy(cave) - 1
            section = section + growth
            section = section % max_level
            section = section + 1

            high, low = i*nrows, (i+1)*nrows
            left, right = j*ncols, (j + 1)*ncols
            res[high:low, left:right] = section

    return res


cave2 = grow_larger_cave(cave)
G2 = build_graph_of_cave(cave2)
enter = (0, 0)
exit = tuple(v - 1 for v in cave2.shape)
path = nx.shortest_path(G2, enter, exit, weight="weight")
risk = sum([cave2[i, j] for i, j in path[1:]])
print(f"Solution to star 1: {risk}.")
