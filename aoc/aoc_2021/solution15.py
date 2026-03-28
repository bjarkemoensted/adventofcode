# ·     . ` ·  `·•.  *`·.  * ·. ` ·.* •   `· . · *. `  ·.   *·  .·  `·.·` *· . `
#  ·`·*. `.·+` *.· `  ·    .  ·` ·    Chiton · *·`·*    · .  `*··.`*.    ··`  ·.
# . · .·`    +·.  `·*. https://adventofcode.com/2021/day/15 .   `  .* ·   `·  `·
# *. ·`  ·• `·. ` ·*.   +.· `  .·* ·`   ·*.  ` .`·*.  · `·  ·   + . · `    `· .·

import copy
from collections import defaultdict
from heapq import heappop, heappush
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

type coordtype = tuple[int, int]
type graphtype = dict[coordtype, dict[coordtype, int]]


def parse(s: str) -> NDArray[np.int_]:
    res = np.array([[int(s) for s in list(line.strip())] for line in s.splitlines()])
    return res


def build_graph_of_cave(cave: NDArray[np.int_]) -> graphtype:
    """Constructs a directional graph of a cave"""
    
    G: graphtype = defaultdict(dict)
    coords_set = set(np.ndindex(cave.shape))

    for v, w in np.ndenumerate(cave):
        i, j = v
        neighbors = ((i+1, j), (i-1, j), (i, j+1), (i, j-1))
        for u in neighbors:
            if u not in coords_set:
                continue
            G[u][v] = int(w)
        #

    return G


@overload
def dijkstra(G: graphtype, source: coordtype, target: coordtype, return_path: Literal[True]) -> list[coordtype]: ...
@overload
def dijkstra(G: graphtype, source: coordtype, target: coordtype, return_path: Literal[False]=False) -> int: ...
def dijkstra(G, source, target, return_path=False) -> int|list[coordtype]:
    """Uses Dijkstra's algorithm to find the shortest path from the source to the target node.
    G: The graph, represented as a dictionary.
    source/target - the source and target nodes.
    return_path (bool, default=False) - whether to return the full path. If False, returns
        the shortest distance (int)"""
    
    # Priority queue for the shortest path in the open set
    queue: list[tuple[int, coordtype]] = []
    # Start with just the source node, at distance 0
    initial_state = (0, source)
    dist = {source: 0}
    heappush(queue, initial_state)

    # The predecessors of each node. For reconstructing the full path
    camefrom: dict[coordtype, coordtype] = dict()

    while queue:
        # If the currently shortest path reaches the target, we're done
        d, u = heappop(queue)
        if u == target:
            # If we don't need the full path, just return the distance
            if not return_path:
                return dist[target]
            
            # Otherwise, reconstruct the path
            node = u
            path_rev = [node]
            while node in camefrom:
                node = camefrom[node]
                path_rev.append(node)
            
            path = path_rev[::-1]
            return path
        
        # Drop this path if we have a faster path to u
        if d > dist.get(u, float("inf")):
            continue
        
        # Otherwise, check all neighbors or u
        for v, delta in G[u].items():
            d_new = d + delta
            improved = d_new < dist.get(v, float("inf"))
            if improved:
                # If this beats the current best path to v, update dist and predecessor info
                dist[v] = d_new
                camefrom[v] = u
                heappush(queue, (d_new, v))
            #
        #
    
    raise RuntimeError("No path found")


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
    star1 = dijkstra(G, enter, exit)
    print(star1)
    #star1 = sum([cave[i, j] for i, j in path[1:]])  # Remember the first step doesn't count
    print(f"Solution to part 1: {star1}")

    cave2 = grow_larger_cave(cave)
    G2 = build_graph_of_cave(cave2)
    enter = (0, 0)
    exit = tuple(v - 1 for v in cave2.shape)
    star2 = dijkstra(G2, enter, exit)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 15
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
