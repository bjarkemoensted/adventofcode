# ·     . ` ·  `·•.  *`·.  * ·. ` ·.* •   `· . · *. `  ·.   *·  .·  `·.·` *· . `
#  ·`·*. `.·+` *.· `  ·    .  ·` ·    Chiton · *·`·*    · .  `*··.`*.    ··`  ·.
# . · .·`    +·.  `·*. https://adventofcode.com/2021/day/15 .   `  .* ·   `·  `·
# *. ·`  ·• `·. ` ·*.   +.· `  .·* ·`   ·*.  ` .`·*.  · `·  ·   + . · `    `· .·

import copy

import networkx as nx
import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> NDArray[np.int_]:
    res = np.array([[int(s) for s in list(line.strip())] for line in s.splitlines()])
    return res


def build_graph_of_cave(cave: NDArray[np.int_]) -> nx.DiGraph:
    """Constructs a directional graph of a cave"""
    G = nx.DiGraph()
    # Connect each node to its neighbors, using neighbor risk as edge weight
    coords = [(i, j) for i, j in np.ndindex(cave.shape)]
    coords_set = set(coords)

    for u in coords:
        i, j = u
        neighbors = ((i+1, j), (i-1, j), (i, j+1), (i, j-1))
        for v in neighbors:
            if v not in coords_set:
                continue

            ip, jp = v
            weight = cave[ip, jp]
            G.add_edge(u, v, weight=weight)
    return G


def grow_larger_cave(cave: NDArray[np.int_], factor=5, max_level=9) -> NDArray[np.int_]:
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
        #

    return res


def solve(data: str) -> tuple[int|str, ...]:
    cave = parse(data)
    G = build_graph_of_cave(cave)
    enter = (0,0)
    exit = tuple(v - 1 for v in cave.shape)
    
    # Find the shortest path through the cave
    path = nx.shortest_path(G, enter, exit, weight="weight")
    star1 = sum([cave[i, j] for i, j in path[1:]])  # Remember the first step doesn't count
    print(f"Solution to part 1: {star1}")

    cave2 = grow_larger_cave(cave)
    G2 = build_graph_of_cave(cave2)
    enter = (0, 0)
    exit = tuple(v - 1 for v in cave2.shape)
    path = nx.shortest_path(G2, enter, exit, weight="weight")
    star2 = sum([cave2[i, j] for i, j in path[1:]])
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 15
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
