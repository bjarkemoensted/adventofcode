# *· `·    +   *  ·` · +.   ·`     · ·. *   ·•.* `   ·    *·`*.  · ·+·     · .`·
# ·` . +.*·`·    .*·  · *.· Many-Worlds Interpretation .  ·•  + ·  .·+ ` ··`  .*
# `•·· `   ·+ · .· *   https://adventofcode.com/2019/day/18  `·+. * ·   ·` * ·`·
#  ·`+.··     *.·`  ·   ·`. + ·  .      · ·.  *` ·    ·*       ·` ·* `    * ·`·.

from collections import defaultdict, deque
from enum import StrEnum
from functools import cache
from heapq import heappop, heappush
from string import ascii_lowercase
from typing import Iterator, NamedTuple, Self

import networkx as nx
import numpy as np
from numpy.typing import NDArray

type coord = tuple[int, int]

# Distinct powers of 2 for each key/door, so we can use bit making for set operations
KEY_MASK = {c: 1 << i for i, c in enumerate(ascii_lowercase)}
DOOR_MASK = {k.upper(): v for k, v in KEY_MASK.items()}


class Symbol(StrEnum):
    SPACE = "."
    WALL = "#"
    ENTRANCE = "@"


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line) for line in s.splitlines()])
    return res


class State(NamedTuple):
    pos: tuple[coord, ...]
    keys: int

    def move(self, worker: int, target: coord, new_key: int=0) -> Self:
        assert 0 <= worker < len(self.pos)
        pos: tuple[coord, ...] = self.pos[:worker] + (target,) + self.pos[worker+1:]

        res = self.__class__(pos=pos, keys=self.keys+new_key)
        return res


class PriorityQueue[T]:
    def __init__(self) -> None:
        self._counter = 0
        self.a: list[tuple[int, int, T]] = []
        self._counts: dict[T, int] = defaultdict(int)
    
    def put(self, elem: T, priority: int) -> None:
        heappush(self.a, (priority, self._counter, elem))
        self._counter += 1
        self._counts[elem] += 1
    
    def get(self) -> T:
        _, _, elem = heappop(self.a)
        self._counts[elem] -= 1
        return elem

    def __contains__(self, item):
        return self._counts[item] > 0

    def __bool__(self):
        return bool(self.a)
    
    def __len__(self) -> int:
        return len(self.a)


def _build_graph(node_coords: set[coord]) -> nx.Graph:
    """Builds a networkx graph from a set of coordinates. Adjacent coordinates are connected."""
    G = nx.Graph()
    
    for u in node_coords:
        i, j = u
        for v in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            if v in node_coords:
                G.add_edge(u, v)
            #
        #

    return G


def make_multi_entrance(M: NDArray[np.str_]) -> NDArray[np.str_]:
    """Turns a vault map into a multi-entrance map, by replacing the region around the entrance
    with a walled area with an entrance in each corner"""

    res = M.copy()
    old_entrance = np.argwhere(res == Symbol.ENTRANCE)
    assert len(old_entrance) == 1
    i, j = map(int, old_entrance[0])
    
    # Fill the area around the entrance with walls
    res[i-1:i+2, j-1:j+2] = Symbol.WALL
    # Add entrances in each corner
    for ip in (i-1, i+1):
        for jp in (j-1, j+1):
            res[ip, jp] = Symbol.ENTRANCE
        #

    return res


class Vault:
    def __init__(self, M: NDArray[np.str_]) -> None:
        self.M = M.copy()
        
        # Determine locations of nodes, keys, doors, and entrances
        chars: dict[coord, str] = {
            (i, j): char.item() for (i, j), char in np.ndenumerate(self.M) if char != Symbol.WALL
        }
        self.nodes = set(chars.keys())
        self.entrance_positions = tuple((i, j) for (i, j), char in chars.items() if char == Symbol.ENTRANCE)
        # Dicts mapping the symbol for each key/door to its coordinate
        self.key_lookup = {char: (i, j) for (i, j), char in chars.items() if char in KEY_MASK}
        self.door_lookup = {char: (i, j) for (i, j), char in chars.items() if char in DOOR_MASK}

        # Run some checks
        assert {s.lower() for s in self.door_lookup.keys()}.issubset(set(self.key_lookup.keys()))
        assert len(self.entrance_positions) > 0

        # Sets of coordinates for keys and doors
        self.key_coords = set(self.key_lookup.values())
        self.door_coords = set(self.door_lookup.values())
        self.keys_order = tuple(sorted(self.key_lookup.keys()))
        self.all_keys = frozenset(self.keys_order)
        self.keys_sum = sum(KEY_MASK[k] for k in self.keys_order)
        
        # Raw graph connecting adjacent nodes
        self.G_raw = _build_graph(self.nodes)

        # Build a graph connecting all keys and entrances
        self.G = nx.Graph()
        nodes_of_note = list(self.entrance_positions) + sorted(self.key_coords)
        for i, u in enumerate(nodes_of_note):
            startat = max(len(self.entrance_positions), i+1)
            dists_from_source = nx.single_source_shortest_path_length(self.G_raw, u)
            for v in nodes_of_note[startat:]:
                if v in dists_from_source:
                    self.G.add_edge(u, v, weight=dists_from_source[v])
        
        # Cache the min distance from each source, for heuristics
        self._dist_to_nearest_neighbor = {
            pos: min(dist for _, v, dist in self.G.edges(pos, data="weight") if v in self.key_coords)
            for pos in self.G.nodes() 
        }

        # For each pair of nodes, stores the shortest path for each subset of doors passed
        self.distinct_connections: dict[coord, dict[coord, dict[int, int]]] = {
            u: self._distinct_connections(u) for u in nodes_of_note
        }
    
    @cache
    def _door_int_to_set(self, doors_int: int) -> set[str]:
        res = {char for char, n in DOOR_MASK.items() if (n & doors_int)}
        return res

    def _distinct_connections(self, u: coord) -> dict[coord, dict[int, int]]:
        """Determines the best connections from node u, for every possible subset
        of door nodes passed."""

        res: dict[coord, dict[int, int]] = {v: dict() for v in self.key_coords}

        # Represent each active path with (head, visited, doors, doors_int)
        paths: deque[tuple[coord, set[coord], int]] = deque()
        initial: tuple[coord, set[coord], int] = (u, {u}, 0)
        paths.append(initial)

        def process(reached: coord, distance_: int, doors_passed: int) -> bool:
            """Update results with the newly found path. Returns a boolean indicating
            whether the path should be continued"""
            
            nonlocal res
            d_ = res[reached]
            # Ignore if an existing requires a subset of keys and has <= length
            inferior = any(
                ((otherdoors_ & doors_passed) == otherdoors_) and otherdist_ <= distance_
                for otherdoors_, otherdist_ in d_.items()
            )
            if inferior:
                return False
            
            res[reached][doors_passed] = distance_
            return True

        while paths:
            head, visited, doors = paths.popleft()
            neighbors = [v for v in self.G_raw.neighbors(head) if v not in visited]
            for v in neighbors:
                i, j = v
                new_doors = doors
                # If forced to move straight, reuse visited set bc set ops are expensive
                new_visited = visited if len(neighbors) == 1 else visited.copy()
                
                # Add new doors when encountered
                if v in self.door_coords:    
                    door_symbol = self.M[i, j].item()
                    new_doors += DOOR_MASK[door_symbol]
                
                # Add to results when finding a new path to a key
                if v in self.key_coords:
                    promising = process(v, len(visited), doors)
                    if not promising:
                        continue
                    #
                
                # Put updated state back on the queue
                new_visited.add(v)
                paths.append((v, new_visited, new_doors))
            #

        return res

    @cache
    def _closest(self, coords: frozenset[coord]) -> int:
        """Lower bound on visiting all input coords, computed as the sum
        of the distance to the nearest neighbor of each node"""
        res = 0
        for u in coords:
            res += min((dist for _, v, dist in self.G.edges(u, data="weight") if v in coords), default=0)

        return res

    def _heuristic_subsets(self, state: State) -> int:
        """Heuristic which considers the subset of remaining relevant node (current
        position and each remaining key). Gives a tighter bound than the naive
        approach but is slower"""
        subset = frozenset({self.key_lookup[k] for k in self._get_missing_keys(state.keys)} | {p for p in state.pos})
        res = self._closest(subset)
        return res

    @cache
    def _heuristic_naive(self, keys: int) -> int:
        """Sums the distance to the nearest neighbor of each remaining key"""
        locs = (self.key_lookup[s] for s in self._get_missing_keys(keys))
        res = sum(self._dist_to_nearest_neighbor[k] for k in locs)
        return res

    def heuristic(self, state: State) -> int:
        """Lower bound on the remaining number of steps.
        When there are multiple entrances, considers only the subset of
        current positions and remaining keys, because the naive approach causes
        a large number of iterations"""
        if len(self.entrance_positions) > 1:
            return self._heuristic_subsets(state)
        else:
            return self._heuristic_naive(state.keys)
        #

    @cache
    def _get_missing_keys(self, key_mask: int) -> tuple[str, ...]:
        res = tuple(k for k in self.keys_order if not (KEY_MASK[k] & key_mask))
        return res

    def get_neighbors(self, state: State) -> Iterator[tuple[int, State]]:
        """Get the neighboring state, i.e. the reachable keys from the position
        of the current state"""

        for mk in self._get_missing_keys(state.keys):
            mask = KEY_MASK[mk]
            v = self.key_lookup[mk]
            for i, u in enumerate(state.pos):
                options = (
                    dist for doors, dist in self.distinct_connections[u][v].items()
                    if ((doors & state.keys) == doors)
                )

                d = min(options, default=-1)
                if d != -1:
                    newstate = state.move(worker=i, target=v, new_key=mask)
                    yield d, newstate
                #
            #
        #

    def shortest_route(self) -> int:
        """A* approach to determining the optimal route"""
        
        s0 = State(pos=self.entrance_positions, keys=0)
        h0 = self.heuristic(s0)
        d_f = {s0: h0}
        d_g = {s0: 0}
        record = float("inf")

        queue: PriorityQueue[State] = PriorityQueue()
        queue.put(s0, priority=h0)

        while queue:
            u = queue.get()
            
            # Done if we have all the keys
            done = u.keys == self.keys_sum
            if done:
                return d_g[u]
            
            for delta, v in self.get_neighbors(u):
                g = d_g[u] + delta
                improved = g < d_g.get(v, float("inf"))
                if improved:
                    d_g[v] = g
                    h = self.heuristic(v)
                    record = min(record, h)
                    f = g + h
                    d_f[v] = f
                    queue.put(v, priority=f)
                #
            #
        raise RuntimeError("No path found")
    #


def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)
    G = Vault(M)

    star1 = G.shortest_route()
    print(f"Solution to part 1: {star1}")

    M_multi = make_multi_entrance(M)
    G2 = Vault(M_multi)

    star2 = G2.shortest_route()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 18
    from aocd import get_data

    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
