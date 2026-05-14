# *· `·    +   *  ·` · +.   ·`     · ·. *   ·•.* `   ·    *·`*.  · ·+·     · .`·
# ·` . +.*·`·    .*·  · *.· Many-Worlds Interpretation .  ·•  + ·  .·+ ` ··`  .*
# `•·· `   ·+ · .· *   https://adventofcode.com/2019/day/18  `·+. * ·   ·` * ·`·
#  ·`+.··     *.·`  ·   ·`. + ·  .      · ·.  *` ·    ·*       ·` ·* `    * ·`·.

from collections import defaultdict, deque
from enum import StrEnum
from functools import cache
from heapq import heappop, heappush
from string import ascii_lowercase, ascii_uppercase
from typing import Iterator, NamedTuple, Self

import networkx as nx
import numpy as np
from numpy.typing import NDArray

type coord = tuple[int, int]


class Symbol(StrEnum):
    SPACE = "."
    WALL = "#"
    ENTRANCE = "@"


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line) for line in s.splitlines()])
    return res


class State(NamedTuple):
    pos: tuple[coord, ...]
    keys: frozenset[str]

    def move(self, worker: int, target: coord, new_key: str|None) -> Self:
        assert 0 <= worker < len(self.pos)
        pos: tuple[coord, ...] = tuple(target if worker == i else p for i, p in enumerate(self.pos))
        keys = self.keys if new_key is None else self.keys | {new_key}
        res = self.__class__(pos=pos, keys=keys)
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
        M = M.copy()

        self.M = M
        self.entrance_positions: tuple[coord, ...] = ()
        self.nodes: set[coord] = set()
        # Dicts mapping the symbol for each key/door to its coordinate
        self.key_lookup: dict[str, coord] = dict()
        self.door_lookup: dict[str, coord] = dict()

        for (i, j), char_np in np.ndenumerate(self.M):
            char = char_np.item()
            if char == Symbol.WALL:
                continue
            
            x = (i, j)
            self.nodes.add(x)
            if char in ascii_lowercase:
                self.key_lookup[char] = x
            elif char in ascii_uppercase:
                self.door_lookup[char] = x
            elif char == Symbol.ENTRANCE:
                self.entrance_positions += (x,)
            else:
                assert char == Symbol.SPACE
            #
        
        # Just make sure we have jeys for all doors
        assert {s.lower() for s in self.door_lookup.keys()}.issubset(set(self.key_lookup.keys()))

        # Sets of coordinates for keys and doors
        self.key_coords = set(self.key_lookup.values())
        self.door_coords = set(self.door_lookup.values())
        self.keys_order = tuple(sorted(self.key_lookup.keys()))
        self.all_keys = frozenset(self.keys_order)
        
        # Raw graph connecting adjacent nodes
        self.G_raw = _build_graph(self.nodes)

        # Graph connecting locations of keys + entrances to each other
        self.G = nx.Graph()
        nodes_of_note = list(self.entrance_positions) + sorted(self.key_coords)

        for i, u in enumerate(nodes_of_note):
            startat = max(len(self.entrance_positions), i+1)
            for v in nodes_of_note[startat:]:
                try:
                    dist = nx.shortest_path_length(self.G_raw, source=u, target=v)
                except nx.NetworkXNoPath:
                    continue
                self.G.add_edge(u, v, weight=dist)
            #
        
        self.distinct_connections: dict[coord, dict[coord, dict[frozenset[str], int]]] = {
            u: self._distinct_connections(u) for u in nodes_of_note
        }

    def _distinct_connections(self, u: coord) -> dict[coord, dict[frozenset[str], int]]:
        """Determines the best connections from node u, for every possible subset
        of door nodes passed."""

        res: dict[coord, dict[frozenset[str], int]] = {v: dict() for v in self.key_coords}

        # Represent each active path with (head, visited, doors)
        paths: deque[tuple[coord, set[coord], set[str]]] = deque()
        initial: tuple[coord, set[coord], set[str]] = (u, {u}, set())
        paths.append(initial)

        def process(reached: coord, distance_: int, doors_passed: set[str]) -> bool:
            """Update results with the newly found path. Returns a boolean indicating
            whether the path should be continued"""
            
            nonlocal res
            # Ignore path if a better one has been found
            d_ = res[reached]
            inferior = any(
                otherdoors_.issubset(doors_passed) and otherdist_ <= distance_
                for otherdoors_, otherdist_ in d_.items()
            )
            if inferior:
                return False
            
            res[reached][frozenset(doors_passed)] = distance_
            return True

        while paths:
            head, visited, doors = paths.popleft()
            for v in self.G_raw.neighbors(head):
                if v in visited:  # Don't go backwards
                    continue
                i, j = v
                # Update state for active path
                new_visited = visited | {v}
                new_doors = doors
                
                # Add new doors when encountered
                if v in self.door_coords:
                    door_symbol = self.M[i, j].item()
                    assert door_symbol.isupper()
                    new_doors = doors | {door_symbol}
                
                # Add to results when finding a new path to a key
                if v in self.key_coords:
                    promising = process(v, len(visited), doors)
                    if not promising:
                        continue
                
                paths.append((v, new_visited, new_doors))
            #

        return res

    @cache
    def _mst(self, nodes: frozenset[coord]) -> int:
        G = nx.subgraph(self.G, nodes)
        mst = nx.minimum_spanning_tree(G)
        res = int(mst.size(weight="weight"))
        return res

    def heuristic(self, state: State) -> int:
        """Lower bound on the minimum distance needed to recover the missing keys"""
        nodes_of_interest = {v for k, v in self.key_lookup.items() if k not in state.keys}
        nodes_of_interest |= set(state.pos)
        res = self._mst(frozenset(nodes_of_interest))
        return res

    def get_neighbors(self, state: State) -> Iterator[tuple[int, State]]:
        """Get the neighboring state, i.e. the reachable keys from the position
        of the current state"""
        missing_keys = self.all_keys - state.keys
        opened = {s.upper() for s in state.keys}

        for i, u in enumerate(state.pos):
            for mk in missing_keys:
                v = self.key_lookup[mk]
                options = (dist for doors, dist in self.distinct_connections[u][v].items() if doors.issubset(opened))
                d = min(options, default=-1)
                if d != -1:
                    newstate = state.move(worker=i, target=v, new_key=mk)
                    yield d, newstate
                #
        #

    def shortest_route(self, maxiter=-1) -> int:
        s0 = State(pos=self.entrance_positions, keys=frozenset())
        h0 = self.heuristic(s0)
        d_f = {s0: h0}
        d_g = {s0: 0}
        record = float("inf")

        queue: PriorityQueue[State] = PriorityQueue()
        queue.put(s0, priority=h0)
        nits = 0

        while queue:
            nits += 1
            if maxiter != -1 and nits > maxiter:
                break
            u = queue.get()
            
            done = u.keys == self.all_keys
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
