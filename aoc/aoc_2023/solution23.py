# ·*·`.  + · + `     ·* .   .` *·  .`·*    ·  +·  `  *. · `. · + .   ·*  .`•·· ·
#   `·  · .   `+· . ·. · ` * ·.    A Long Walk .`.*·      · .`+  · .· +*`·   .· 
# *· . ·   `.  ·.•·    https://adventofcode.com/2023/day/23   ·   .*  ·.* ·`  ·.
# ·`*    ··.`·.*      ·`·.*`·*   .·`.·  +·.      · `·.*    `   · ` ·    ·*.·.`+·

from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
import numba
from numba.typed import Dict, List
from numba.types import DictType
import numpy as np
from numpy.typing import NDArray
import time
import typing as t


coordtype: t.TypeAlias = tuple[int, int]
edgetype: t.TypeAlias = tuple[coordtype, coordtype]
graphtype: t.TypeAlias = dict[coordtype, dict[coordtype, int]]

nb_node = numba.types.uint64
nb_inner = numba.types.DictType(nb_node, nb_node)
nb_outer = numba.types.DictType(nb_node, nb_inner)


class Symbols(str, Enum):
    wall = "#"
    free = "."
    up = "^"
    right = ">"
    down = "v"
    left = "<"


raw = """#.#####################
#.......#########...###
#######.#########.#.###
###.....#.>.>.###.#.###
###v#####.#v#.###.#.###
###.>...#.#.#.....#...#
###v###.#.#.#########.#
###...#.#.#.......#...#
#####.#.#.#######.#.###
#.....#.#.#.......#...#
#.#####.#.#.#########v#
#.#...#...#...###...>.#
#.#.#v#######v###.###v#
#...#.>.#...>.>.#.###.#
#####v#.#.###v#.#.###.#
#.....#...#...#.#.#...#
#.#########.###.#.#.###
#...###...#...#...#.###
###.###.#.###v#####v###
#...#...#.#.>.>.#.>.###
#.###.###.#.###.#.#v###
#.....###...###...#...#
#####################.#"""


up = (-1, 0)
right = (0, 1)
down = (1, 0)
left = (0, -1)

slopes = {
    up: Symbols.up,
    right: Symbols.right,
    down: Symbols.down,
    left: Symbols.left,
}

_all_dirs: tuple[tuple[int, int], ...] = tuple(sorted(slopes.keys()))


def parse(s: str) -> NDArray[np.str_]:
    M = np.array([list(line) for line in s.split("\n")])
    return M


def _dirs_and_neighbors(M: NDArray[np.str_], i: int, j: int) -> t.Iterator[tuple[tuple[int, int], coordtype]]:
    """For a given site in the ASCII map, generates pairs of (direction, neighbor coordinates)."""

    for dir_ in slopes.keys():
        di, dj = dir_
        x = (i + di, j + dj)
        if all(0 <= c < lim for c, lim in zip(x, M.shape)) and M[*x] != Symbols.wall:
            yield dir_, x
        #
    #


def _iter_segments(M: NDArray[np.str_]) -> t.Iterator[tuple[tuple[coordtype, ...], bool]]:
    """Given the ASCII map, generates every path segment, meaning all paths between non-trivial sites (with != 2 neighbors).
    Generates for each such segment
    points on path - tuple of the sites that make up the path segment,
    uphill (bool) - indicates whether any point on the segment goes up a slope"""

    sites = {x for x, char in np.ndenumerate(M) if char != Symbols.wall}
    _adj_cache = {(i, j): tuple(_dirs_and_neighbors(M, i, j)) for (i, j) in sites}

    branch_points = {x for x, adj_ in _adj_cache.items() if len(adj_) != 2}

    for x in sorted(branch_points):
        paths = [((x,), False)]
        visited = {x}

        while paths:
            grow = paths
            paths = []
            for points, uphill in grow:
                head = points[-1]
                for step, neighbor in _adj_cache[head]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)

                    char = M[*head]
                    uphill_here = char != Symbols.free and slopes[step] != char
                    new_uphill = uphill or uphill_here
                    
                    new_points = points + (neighbor,)
                    updated_path = (new_points, new_uphill)

                    if neighbor in branch_points:
                        yield updated_path
                    else:
                        paths.append(updated_path)
                    #
                #
            #
        #
    #


def build_graph(M: NDArray[np.str_], allow_uphill=False) -> graphtype:
    G: graphtype = dict()

    for path, uphill in _iter_segments(M):
            if allow_uphill or not uphill:
                dist = len(path) - 1
                endpoints = path[0], path[-1]
                for node in endpoints:
                    if node not in G:
                        G[node] = dict()
                    #
                u, v = endpoints
                G[u][v] = dist
            #
        #

    return G


def heuristic(G: dict[int, dict[int, int]], visited: int, head: int):
    return 99999  # !!!


def astar(G: dict[int, dict[int, int]], nodes, edges, keys, start: int, end: int, n_max: int=-1) -> int:


    # Priority queue storing each state (tuple of key for nodes visited and current head)
    queue: list[tuple[int, tuple[int, int]]] = []

    initial_visited = keys[start]
    initial_state = (initial_visited, start)
    h0 = heuristic(G, initial_visited, start)
    heappush(queue, (h0, initial_state))

    d_g = {(i, i): i for i in range(0)}
    d_g[initial_state] = 0

    nits = 0
    record = -1

    while queue and (nits < n_max or n_max == -1):
        nits += 1

        h_old_neg, state = heappop(queue)
        visited, head = state
        h_old = -h_old_neg
        dist = d_g[state]

        if head == end:
            record = max(record, dist)
            continue

        for new_head, delta in G[head].items():
            if visited & new_head:
                continue

            new_dist = dist + delta
            new_visited = visited + keys[new_head]
            new_state = (new_visited, new_head)

            if new_dist <= d_g.get(new_state, -1):
                continue
            
            d_g[new_state] = new_dist
            upper_bound = new_dist + heuristic(G, new_visited, new_head)
            heappush(queue, (-upper_bound, new_state))
            
        
        print(f"!!! {visited=}, {head=}")

    return record


def longest_path(G_coord: graphtype, start_coord: coordtype, end_coord: coordtype) -> int:
    
    # Map the nodes in (i, j) coordinates onto consecutive inds starting at 0
    _nodes_coord = sorted(G_coord.keys())
    inv = {coord: i for i, coord in enumerate(_nodes_coord)}
    nodes = [inv[u] for u in _nodes_coord]
    G = {inv[u]: {inv[v]: dist for v, dist in d.items()} for u, d in G_coord.items()}

    assert len(_nodes_coord) < 64 - 1
    keys = [2**i for i in range(len(_nodes_coord))]

    # Get all the distinct edges (regardless of direction)
    _edge_lookup = {tuple(sorted((u, v))): dist for u, d in G.items() for v, dist in d.items()}
    edges_by_size = sorted(_edge_lookup.items(), key=lambda t: t[-1])

    start, end = inv[start_coord], inv[end_coord]
    
    res = astar(G, nodes, edges_by_size, keys, start, end, n_max=5)   # !!!

    return res

    # !!!
    nodes_ordered = sorted(G.nodes())
    inv = {node: i for i, node in enumerate(nodes_ordered)}
    stuff = [((-1, -1), (-1, -1), (-1, -1), (-1, -1)) for _ in nodes_ordered]

    for u_i, u in enumerate(nodes_ordered):
        adj = [(-1, -1) for _ in range(4)]
        for ind, (v, dist) in enumerate(sorted(G.d[u].items())):
            adj[ind] = (inv[v], dist)
        
        a, b, c, d = adj
        stuff[u_i] = (a, b, c, d)


    pre = time.time()
    res = astar(stuff, inv[start], inv[end])

    print(f"Delta: {time.time() - pre:.6f}")

    return res


def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)
    G = build_graph(M)
    G2 = build_graph(M, allow_uphill=True)

    start = min(G.keys())
    end = max(G.keys())

    
    # TODO solve puzzle
    star1 = longest_path(G, start, end)
    print(f"Solution to part 1: {star1}")

    
    star2 = longest_path(G2, start, end)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 23
    from aocd import get_data
    #raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
