# • ·  `   . ·  *·  .`* ·    ·   *•· · * `+·.` ·  *  `    ` ··.` *·+ `· . `  *·.
# `*  .· ` ·    · +`       `.  Air Duct Spelunking    .·   ·  * · . `* ·    `·.*
# .`. * ··· `   .  *·  https://adventofcode.com/2016/day/24    ·`  · · `* ··  *·
# ··`*· `. +   ·  · *`       ·.`· `*·•.  ·  `    ··       +`  · .   + `.·· +· ` 


from functools import cache
import heapq
import networkx as nx


def parse(s: str):
    """Builds a 2D-lattice where all non-wall neighboring sites are connected"""
    G = nx.Graph()
    lines = [list(line.strip()) for line in s.split("\n")]
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == "#":
                continue
            u = (i, j)
            neighbors = ((i+1, j), (i, j+1))
            for ii, jj in neighbors:
                v = (ii, jj)
                try:
                    neighbor_char = lines[ii][jj]
                except IndexError:
                    continue
                if neighbor_char == "#":
                    continue
                G.add_edge(u, v)
            if char != ".":
                # Store points of interest (with an integer on the map) as node properties
                n = int(char)
                G.nodes[u]["n"] = n
            #
        #
    return G


def build_shortest_paths_graph(lattice: nx.Graph) -> nx.Graph:
    """Builds a graph where each pair of nodes is connected with weight given by the shortest distance between the
    two nodes."""

    G = nx.Graph()
    d = dict(nx.get_node_attributes(lattice, "n"))
    nodes = sorted(d.items())
    for i, (u, n_u) in enumerate(nodes):
        for v, n_v in nodes[i+1:]:
            shortest_path = nx.shortest_path(lattice, source=u, target=v)
            # Only store paths directly connecting two points of interest. Paths through other POIs are ignored
            poi_nodes = [lattice.nodes[node].get("n") for node in shortest_path]
            path_is_direct = sum(node is not None for node in poi_nodes) == 2
            if path_is_direct:
                G.add_edge(n_u, n_v, weight=len(shortest_path) - 1)
        #

    return G


@cache
def _path_as_tuple(path: tuple) -> tuple:
    """Represents a path, such as (0, 4, 2, 1) as a tuple of visited nodes (sorted numerically) and the current node,
    e.g. ((0, 2, 4), 1)"""

    head = path[-1]
    tail = path[:-1]
    res = (tuple(sorted(set(tail))), head)
    return res


class PathCache:
    def __init__(self):
        """Helper object for keeping track of the most effective paths for reaching a given state.
        State here is the relevant properties of a path meaning 1) the set of nodes visited and 2) the current location.
        This is because paths like (1,2,3,4) and (1,3,2,4) are equivalent when searching for shortest paths."""

        self._path2dist = dict()  # Keeps track of path lengths
        self._state2shortest = dict()  # Maps states to the shortest distance in which the state has been reached

    def new_shortest(self, path: tuple, dist: int):
        """Takes a path and distance. Determines if the path sets a new record for shortest way to reach its state.
        Registers the path and state with the distance in the cache in the process, and returns a boolean indicating
        whether the path is a new best to the state."""

        self._path2dist[path] = dist
        state = _path_as_tuple(path=path)
        shorter = dist < self._state2shortest.get(state, float("inf"))
        if shorter:
            self._state2shortest[state] = dist
        return shorter

    def __getitem__(self, path):
        return self._path2dist.__getitem__(path)


def _lower_bounds_from_n_remaining(G: nx.Graph) -> dict:
    """Maps number of missing nodes in a path to a lower bound on the number of steps required to visit all nodes."""

    dists_desc = sorted([d["weight"] for _, _, d in G.edges(data=True)], reverse=True)
    res = {0: 0}  # If all nodes have been visited, the lower bound of the remaining path is zero
    for n_remaining in range(1, len(G.nodes()) + 1):
        # Otherwise, add the next smallest distance to the lower bound with n-1 nodes unvisited
        bound = res[n_remaining - 1] + dists_desc.pop()
        res[n_remaining] = bound

    return res


def ts_solve(G: nx.Graph, startat=0, return_to_starting_point=False, maxiter=None) -> int:
    """A* kinda approach to the Traveling Salesman problem."""
    if maxiter is None:
        maxiter = float("inf")

    all_nodes = set(G.nodes())
    d_lower = _lower_bounds_from_n_remaining(G=G)

    # Make a cache and register the initial path, consisting only of the starting node
    cache = PathCache()
    initial_path = (startat,)
    cache.new_shortest(path=initial_path, dist=0)

    # Queue paths prioritized by their heuristic (lower bound on final path length)
    queue = [(d_lower[len(all_nodes)], initial_path)]
    heapq.heapify(queue)

    n_its = 0
    while queue and n_its < maxiter:
        _, path = heapq.heappop(queue)
        visited_all_nodes = set(path) == all_nodes
        correct_loc = path[-1] == startat or not return_to_starting_point
        if visited_all_nodes and correct_loc:
            dist = cache[path]
            return dist

        head = path[-1]
        for v in G.neighbors(head):
            # Grow path with each neighbor in the graph
            d_uv = G[head][v]["weight"]
            dist = cache[path] + d_uv
            new_path = path + (v,)
            if cache.new_shortest(path=new_path, dist=dist):
                # If the new path is a new best way to reach its state, add back to queue
                n_remaining = len(all_nodes - set(new_path))
                heuristic = d_lower[n_remaining]
                priority = dist + heuristic
                heapq.heappush(queue, (priority, new_path))
            #
        n_its += 1
    
    raise RuntimeError("No path found")


def solve(data: str) -> tuple[int|str, int|str]:
    lattice = parse(data)
    G = build_shortest_paths_graph(lattice)

    star1 = ts_solve(G=G, maxiter=None)
    print(f"The shortest path visiting all nodes requires {star1} steps.")

    star2 = ts_solve(G=G, return_to_starting_point=True, maxiter=None)
    print(f"The shortest path with the robot returning requires {star2} steps.")

    return star1, star2


def main() -> None:
    year, day = 2016, 24
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()