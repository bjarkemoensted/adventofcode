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


# TODO fuck it just nuke this!!!!!
class Graph:
    def __init__(self, M: NDArray[np.str_], allow_uphill=False) -> None:
        self.M = M.copy()
        self.d: dict[coordtype, dict[coordtype, int]] = dict()
        self._points: dict[edgetype, tuple[coordtype, ...]] = dict()

        for path, uphill in _iter_segments(M):
            if allow_uphill or not uphill:
                self.add_path(path)
            #
        #

    def add_node(self, u: coordtype) -> None:
        if u not in self.d:
            self.d[u] = dict()
        #
    
    def add_edge(self, u: coordtype, v: coordtype, dist: int) -> None:
        for node in (u, v):
            self.add_node(node)
        
        self.d[u][v] = dist
    
    def add_path(self, path: tuple[coordtype, ...]) -> None:
        dist = len(path) - 1
        u, v = path[0], path[-1]
        assert u != v
        self.add_edge(u, v, dist)
        self._points[(u, v)] = path

    def nodes(self) -> list[coordtype]:
        return sorted(self.d.keys())

    def edges(self) -> list[edgetype]:
        return sorted((u, v) for u, d_ in self.d.items() for v, _ in d_.items())
    #


@numba.njit
def astar(graph: list[tuple[tuple[int, int], ...]], start: int, end: int):
    nodes = list(range(len(graph)))
    n_nodes = len(graph)

    # Binary repr of each node has a single 1 in a distinct location. This allows efficient subset detection
    assert n_nodes < 64 - 1
    keys = [2**i for i in range(n_nodes)]
    for i in range(len(keys)-1):
        # Make sure we don't get overflow errors
        assert keys[i] < keys[i+1]

    # Absolute upper bound heuristic. This is terrible, just need something that's guaranteed to overshoot rn
    upper_abs = 999
    for e in graph:
        for v, delta in e:
            if delta != -1:
                upper_abs += delta
            #
        #

    dg = {(i, i): i for i in range(0)}

    n_max = -1
    initial_key = keys[start]

    # map (head, key) for current longest dist
    dg[(start, initial_key)] = 0

    seed = (initial_key, start)
    queue = [(-upper_abs, seed)]
    nits = 0

    record = -1

    while queue and (n_max == -1 or nits <= n_max):
        nits += 1

        if nits % 1_000_000 == 0:
            print(f"Queue size: {len(queue)} (nits={nits})")

        h_old, (key, head) = heappop(queue)
        h_old *= -1
        dist = dg[(head, key)]

        if head == end:
            record = max(record, dist)
            continue

        # get neighbors
        for neighbor, delta in graph[head]:
            if neighbor == -1:
                break
            
            if keys[neighbor] & key:
                continue

            new_dist = dist + delta
            new_key = key + keys[neighbor]

            improved = new_dist > dg.get((neighbor, new_key), -1)
            # TODO actually compute something here!!!
            new_upper_bound = upper_abs
            f = new_dist + new_upper_bound

            if improved and f > record:
                dg[(neighbor, new_key)] = new_dist

                
                
                new_state = (new_key, neighbor)
                heappush(queue, (-f, new_state))
            #
        #
    
    print(keys)
    return record


def longest_path(G: Graph, start: coordtype, end: coordtype) -> int:
    # Represent nodes as consecutive ints from 0, and the graph as list of tuples of (neighbor, distance)
    # This is more efficient with numba because of reasons!
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
    G = Graph(M)
    start = min(G.nodes())
    end = max(G.nodes())

    
    # TODO solve puzzle
    star1 = longest_path(G, start, end)
    print(f"Solution to part 1: {star1}")

    G2 = Graph(M, allow_uphill=True)
    star2 = longest_path(G2, start, end)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
