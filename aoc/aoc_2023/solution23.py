# ·*·`.  + · + `     ·* .   .` *·  .`·*    ·  +·  `  *. · `. · + .   ·*  .`•·· ·
#   `·  · .   `+· . ·. · ` * ·.    A Long Walk .`.*·      · .`+  · .· +*`·   .· 
# *· . ·   `.  ·.•·    https://adventofcode.com/2023/day/23   ·   .*  ·.* ·`  ·.
# ·`*    ··.`·.*      ·`·.*`·*   .·`.·  +·.      · `·.*    `   · ` ·    ·*.·.`+·

from enum import Enum
from heapq import heappop, heappush
from numba import njit
from numba.typed import Dict
from numba import types
import numpy as np
from numpy.typing import NDArray
import typing as t


coordtype: t.TypeAlias = tuple[int, int]
edgetype: t.TypeAlias = tuple[coordtype, coordtype]
graphtype: t.TypeAlias = dict[coordtype, dict[coordtype, int]]

nb_node = types.int64
nb_inner = types.DictType(nb_node, nb_node)
nb_outer = types.DictType(nb_node, nb_inner)


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


@njit(cache=True)
def heuristic(
        neighbors: NDArray[np.int64],
        edges: NDArray[np.int64],
        keys: NDArray[np.int64],
        visited: int,
        head: int,
        end: int):
    """Heuristic for determining and upper bound on the longest possible path from the current path head
    to the target (end) node, avoiding nodes already visited.
    Nodes visited are represented by a single large integer."""
    
    visited_running = visited
    blop_sig = 0
    front = [head]
    
    # Run BFS on remaining nodes, using hte path head as the source node.
    while front:
        iter_ = [elem for elem in front]
        front = []
        for u in iter_:
            blop_sig += keys[u]

            for v in neighbors[u]:
                if v == -1 or (keys[v] & visited_running):
                    continue

                visited_running += keys[v]
                front.append(v)
            #
        #
    
    # If there's not path from the current path head to target, no ptah is possible
    target_reachable = keys[end] & blop_sig
    if not target_reachable:
        return -1
    
    # Sum all the edges present in the reachable remaining nodes
    res = 0
    for i, j, dist in edges:
        # Require both nodes are reachable
        if (keys[i] & blop_sig) and (keys[j] & blop_sig):
            res += dist
        #

    return res


@njit(cache=True)
def astar_long(
        adj: NDArray[np.int64],
        neighbors: NDArray[np.int64],
        edges: NDArray[np.int64],
        keys: NDArray[np.int64],
        start: int,
        end: int,
        n_max: int=-1) -> int:
    """A* variant for determining longest path.
    adj is the Adjacency matrix with A[i, j] = distance(i, j).
    neighbours is a n_nodes x 4 array where each of the 4 elements of arr[u] are adjacent nodes or -1
    edges is an n_edges x 3 array with u, v, distance, ordered by decending distance, and only for u < v
    keys is an array of distinct 'keys' (powers of 2) which can be summed to effectively represent a set.
    start and end are the source and target nodes.
    n_max is an optional limit on the number of iterations.
    Returns the longest path length if found, -1 if not found (including if the iteration limit is reached)"""

    # Priority queue storing each state (tuple of key for nodes visited and current head)
    queue: list[tuple[int, tuple[int, int]]] = [(i, (i, i)) for i in range(0)]

    # Define initial state
    initial_visited = keys[start]
    initial_state = (initial_visited, start)
    h0 = heuristic(neighbors, edges, keys, initial_visited, start, end)

    # Add the initial state to the queue
    heappush(queue, (h0, initial_state))
    d_g = {(i, i): i for i in range(0)}
    d_g[initial_state] = 0

    nits = 0
    record = -1

    while queue:
        nits += 1
        if nits % 100_000 == 0:
            print(f"Queue size: {len(queue)}, n iterations: {nits} (record: {record})")
        
        if nits < n_max:
            return -1
        
        _, state = heappop(queue)
        visited, head = state
        dist = d_g[state]

        if head == end:
            record = max(record, dist)
            continue
        
        for new_head in neighbors[head]:
            if new_head == -1:
                continue  # hit a 'fake' neighbor because head has fewer than 4 neighbors
            
            delta = adj[head, new_head]
            
            key = int(keys[new_head])
            if visited & key:
                continue

            new_dist = dist + delta
            new_visited = visited + key
            new_state = (new_visited, new_head)

            if new_dist <= d_g.get(new_state, -1):
                continue
            
            d_g[new_state] = new_dist

            h = heuristic(neighbors, edges, keys, new_visited, new_head, end)
            if h == -1:
                continue

            upper_bound = new_dist + h
            if upper_bound <= record:
                continue
            heappush(queue, (-upper_bound, new_state))

    return record


def longest_path(G_coord: graphtype, start_coord: coordtype, end_coord: coordtype) -> int:
    # Map the nodes in (i, j) coordinates onto consecutive inds starting at 0
    _nodes_coord = sorted(G_coord.keys())
    inv = {coord: i for i, coord in enumerate(_nodes_coord)}
    nodes = [inv[u] for u in _nodes_coord]
    n_nodes = len(nodes)
    assert n_nodes < 64 - 1
    assert nodes[0] == 0 and all(nodes[i+1] == nodes[i]+1 for i in range(len(nodes)-1))  # !!!

    G = {inv[u]: {inv[v]: dist for v, dist in d.items()} for u, d in G_coord.items()}

    # Each node can have max 4 neighbors. Store them in Nx4 matrix for efficiency (using -1 for no neighbor)
    neighbors = np.full((n_nodes, 4), -1, dtype=np.int64)
    
    # Adjacency matrix for the graph
    adj = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    # Distinct key (power of 2) for each node, to facilitate bit masking as lookup
    keys = np.array([2**i for i in range(n_nodes)], dtype=np.int64)

    # Store edges lengths under ordered (u, v) tuples, to avoid double counting
    _edge_lookup: dict[tuple[int, int], int] = dict()

    for u, d in G.items():
        for ind, (v, dist) in enumerate(sorted(d.items())):
            adj[u, v] = dist
            neighbors[u, ind] = v

            edgekey = (u, v) if v > u else (v, u)
            _edge_lookup[edgekey] = dist
        #

    # Store edges in an array of (u, v, dist), descending order of distance
    edges_desc = np.full((len(_edge_lookup), 3), 0, dtype=np.int64)
    for i, ((u, v), dist) in enumerate(sorted(_edge_lookup.items(), key=lambda t: -t[-1])):
        edges_desc[i, 0] = u
        edges_desc[i, 1] = v
        edges_desc[i, 2] = dist
    
    
    start, end = inv[start_coord], inv[end_coord]

    res = astar_long(adj=adj, neighbors=neighbors, edges=edges_desc, keys=keys, start=start, end=end)

    return res


def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)
    G = build_graph(M)
    G2 = build_graph(M, allow_uphill=True)

    start = min(G.keys())
    end = max(G.keys())

    star1 = longest_path(G, start, end)
    print(f"Solution to part 1: {star1}")

    
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
