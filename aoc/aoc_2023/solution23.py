# ·*·`.  + · + `     ·* .   .` *·  .`·*    ·  +·  `  *. · `. · + .   ·*  .`•·· ·
#   `·  · .   `+· . ·. · ` * ·.    A Long Walk .`.*·      · .`+  · .· +*`·   .· 
# *· . ·   `.  ·.•·    https://adventofcode.com/2023/day/23   ·   .*  ·.* ·`  ·.
# ·`*    ··.`·.*      ·`·.*`·*   .·`.·  +·.      · `·.*    `   · ` ·    ·*.·.`+·

import typing as t
from enum import Enum
from heapq import heappop, heappush

import numpy as np
from numba import njit
from numpy.typing import NDArray

coordtype: t.TypeAlias = tuple[int, int]
graphtype: t.TypeAlias = dict[int, dict[int, tuple[int, bool]]]


class Symbols(str, Enum):
    wall = "#"
    free = "."
    up = "^"
    right = ">"
    down = "v"
    left = "<"


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


def parse(s: str) -> NDArray[np.str_]:
    M = np.array([list(line) for line in s.split("\n")])
    return M


def _dirs_and_neighbors(M: NDArray[np.str_], i: int, j: int) -> list[tuple[tuple[int, int], coordtype]]:
    """For a given site in the ASCII map, generates pairs of (direction, neighbor coordinates)."""

    res: list[tuple[tuple[int, int], coordtype]] = []
    for dir_ in slopes.keys():
        di, dj = dir_
        x = (i + di, j + dj)
        if all(0 <= c < lim for c, lim in zip(x, M.shape)) and M[*x] != Symbols.wall:
            res.append((dir_, x))
        #
    return res


def build_graph(M: NDArray[np.str_]) -> dict[int, dict[int, tuple[int, bool]]]:
    """Summarizes the ASCII map into a graph - format: {u: {v1: (dist1, uphill1), ...}, ...}"""
    
    # Find the 'interesting' points (start+end points, and points where a path may branch out)
    adj_cache = {(i, j): _dirs_and_neighbors(M, i, j) for (i, j), char in np.ndenumerate(M) if char != Symbols.wall}
    junctions = sorted((i, j) for (i, j), steps in adj_cache.items() if len(steps) != 2)
    # Order and map to ints 0, 1, ...
    map_ = {coord: i for i, coord in enumerate(junctions)}

    G: dict[int, dict[int, tuple[int, bool]]] = {u: dict() for u in map_.values()}

    # Run BFS from each junction point, keep trak of whether we step up a slope
    for x, u in map_.items():
        # Represent paths with (set of nodes in path, head node, has stepped uphille)
        paths: list[tuple[set[coordtype], coordtype, bool]] = [({x}, x, False)]

        while paths:
            grow = paths
            paths = []
            for visited, head, uphill in grow:
                for step, neighbor in adj_cache[head]:
                    if neighbor in visited:
                        continue

                    char = M[*head]
                    uphill_here = char != Symbols.free and slopes[step] != char
                    new_uphill = uphill or uphill_here

                    try:
                        v = map_[neighbor]
                        dist = len(visited)
                        G[u][v] = (dist, new_uphill)
                    except KeyError:
                        updated_path = (visited | {neighbor}, neighbor, new_uphill)
                        paths.append(updated_path)
                    #
                #
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
    # Keep track of the set of nodes still reachable
    reachable = 0
    front = [head]
    
    # Run BFS on remaining nodes, using hte path head as the source node.
    while front:
        iter_ = [elem for elem in front]
        front = []
        for u in iter_:
            reachable += keys[u]

            for v in neighbors[u]:
                if v == -1 or (keys[v] & visited_running):
                    continue

                visited_running += keys[v]
                front.append(v)
            #
        #
    
    # If there's not path from the current path head to target, no ptah is possible
    target_reachable = keys[end] & reachable
    if not target_reachable:
        return -1
    
    # Sum all the edges present in the reachable remaining nodes
    res = 0
    for i, j, dist in edges:
        # Require both nodes are reachable
        if (keys[i] & reachable) and (keys[j] & reachable):
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

    # Cache for upper bounds for path length given a state
    d_f = {(i, i): i for i in range(0)}

    nits = 0
    # Keep running record of the longest path reaching the target (independent of subset nodes visited)
    record = -1

    while queue:
        nits += 1
        if nits < n_max:
            return -1
        
        # Pop the currently most promising path
        _, state = heappop(queue)
        visited, head = state
        dist = d_g[state]

        # If we're done, update the record for longest path
        if head == end:
            record = max(record, dist)
            continue
        
        # Look at the current path head node's neighbors
        for new_head in neighbors[head]:
            if new_head == -1:
                break  # stop when e run out of neighbors
            
            # Ignore neighbor if it's already in this path
            key = int(keys[new_head])
            if visited & key:
                continue
            
            # Update distance and visited notes
            delta = adj[head, new_head]
            new_dist = dist + delta
            new_visited = visited + key
            new_state = (new_visited, new_head)

            # Don't add the neighbor if it doesn't exceed the current most promising identical state
            if new_dist <= d_g.get(new_state, -1):
                continue
            
            # Compute heuristic for the state
            if new_state in d_f:
                h = d_f[new_state]
            else:
                h = heuristic(neighbors, edges, keys, new_visited, new_head, end)
                d_f[new_state] = h

            # Don't add to path if the target has become unreachable
            if h == -1:
                continue
            
            # Compute new upper bound and check if path is still feasible
            d_g[new_state] = new_dist
            upper_bound = new_dist + h
            if upper_bound <= record:
                continue

            # Put back on the queue
            heappush(queue, (-upper_bound, new_state))

    return record


def longest_path(G: graphtype, start: int, end: int, allow_uphill=False) -> int:
    """Findes the longest path from the input start to end node on the input graph.
    It's assumed that the start and end nodes only connect to one other node in the graph.
    allow_uphill specifies whether stepping uphill on a slope is allowed.
    A number of preprossing steps are done to make the algorithm more efficient:
    * The adjacency matrix for the graph is stored in a numpy array (for looking up distances
    when we have nodes u and v). This is because dicts seem to be kind of slow with numba.
    * The neighbors for each node is stored in another, smaller array. This is because the graph is
    embedded in a lattice, so each node can have max 4 neighbors. Therefore, we store information on
    neighboring node in an Nx4 array, where Arr[u] have 4 values which are the neighbors of u. If u have fewer
    than 4 neighbors, the unused elements are set to -1.
    * We store in another array the distances of all connections between pairs of nodes in the graph, along with the
    two connected nodes. This is for computing upper bounds on the remaining distance more efficiently:
    If the nodes u and v are still unused, only one of the edges u -> v and v -> u may be used. Computing the
    undirected edges in a preprocessing step enables fast computation of the sum of edges between the remaining
    nodes.
    * Finally, we define a 'key' 2^u for each node. This is because unlike in standard pathfinding algorithms like A*,
    we need to keep track of not just one (shortest, in standard A*) path for each node reached, but the longest
    path for each distinct subset visited on paths to the node. We also need to be able to use said subsets to
    efficiently look up the corresponding distances. This is a problem because normal sets aren't hashable, and numba
    functions don't support frozensets (which are). The 'keys' are a workaround for this issue - summing the keys for
    any subset of nodes results in a distinct integer which 1) is hashable and thus can be used as a dict key, and 2)
    allows quick computation of set intersection (bitwise AND)."""

    nodes = sorted(G.keys())
    n_nodes = len(nodes)
    assert all(node == i for i, node in enumerate(nodes))

    # We represent the set of nodes visited by individual bits in a large int, so n_nodes better be small enough
    assert n_nodes < 64 - 1

    # Each node can have max 4 neighbors. Store them in Nx4 matrix for efficiency (using -1 for no neighbor)
    neighbors = np.full((n_nodes, 4), -1, dtype=np.int64)
    
    # Adjacency matrix for the graph
    adj = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    
    # Distinct key (power of 2) for each node, to facilitate bit masking as lookup
    keys = np.array([2**i for i in range(n_nodes)], dtype=np.int64)

    # Store edges lengths under ordered (u, v) tuples, to avoid double counting
    _edge_lookup: dict[tuple[int, int], int] = dict()

    # Populate adjacency, neighbors, and edge data
    for u, d in G.items():
        # Only keep edges with uphill steps if we allow that
        edges_keep = ((v, dist) for v, (dist, uphill) in d.items() if allow_uphill or not uphill)
        for ind, (v, dist) in enumerate(sorted(edges_keep)):
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

    res = astar_long(adj=adj, neighbors=neighbors, edges=edges_desc, keys=keys, start=start, end=end)

    return res


def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)
    G = build_graph(M)
    start = min(G.keys())
    end = max(G.keys())

    star1 = longest_path(G, start, end)
    print(f"Solution to part 1: {star1}")
    
    star2 = longest_path(G, start, end, allow_uphill=True)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
