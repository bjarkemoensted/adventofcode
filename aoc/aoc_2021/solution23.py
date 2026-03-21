# ·.`·*`·   •.· · `*   ·  `· . ·*`.·    ·. •· · ` *   +· `  . `· . +· * •· `.· ·
# *··`. .*  ·    ·  `  ` .· •   · *· Amphipod ` *·.+ `·. *·` ·   ·.`    ` ·· .+*
# ·.*`··*   `·  .  ··  https://adventofcode.com/2021/day/23   + ·*`  · `·.+ · `.
# `·.    ··.*     *`  ·. ·  `·*`.   · ·* ·+`   .·.`   `*· ·   ·   · . ·* ·*   ·`

import itertools
import operator
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cache
from heapq import heappop, heappush
from typing import Iterator, Literal, TypeAlias, TypeGuard, get_args

import networkx as nx
import numpy as np
from numpy.typing import NDArray

extra_snippet = (
    "#D#C#B#A#\n"
    "#D#B#A#C#"
)
  

amphtype: TypeAlias = Literal["A", "B", "C", "D"]
amph_symbols_ordered = get_args(amphtype)
amph_symbols = set(amph_symbols_ordered)


def is_amph(obj) -> TypeGuard[amphtype]:
    return obj in amph_symbols


movement_costs: dict[amphtype, int] = {"A": 1, "B": 10, "C": 100, "D": 1000}


class symbols:
    blank = " "
    empty = "."
    wall = "#"
    amber: amphtype = "A"
    bronze: amphtype = "B"
    copper: amphtype = "C"
    desert: amphtype = "D"


def parse(s: str) -> NDArray[np.str_]:
    lines = s.splitlines()
    width = max(map(len, lines))
    res = np.array([list(line+" "*(width - len(line))) for line in lines])
    return res


def _get_neighbor_coords(*coords: tuple[int, int]) -> set[tuple[int, int]]:
    res = set()
    for i, j in coords:
        res |= {(i+1, j), (i-1, j), (i, j+1), (i, j-1)}

    return res


@dataclass(frozen=True, order=True)
class Amphipod:
    pos: int
    kind: amphtype
    moved_to_hallway: bool=False
    moved_to_room: bool=False


# For storing amphipod states in a hash structure
keytype: TypeAlias = tuple[Amphipod, ...]


def build_graph[T](*coords: tuple[int, int], mapper: dict[tuple[int, int], T]) -> nx.Graph:
    """Builds an undirected graph representing the input coordinates, with neighboring coordinates linked.
    mapper is a dictionary mapping coordinate tuples to whatever is to represent nodes in the
    final graph."""

    coords_set = set(coords)
    G = nx.Graph()

    for i, j in coords:
        u = mapper[(i, j)]
        for v_coord in _get_neighbor_coords((i, j)):
            if v_coord in coords_set:
                v = mapper[v_coord]
                G.add_edge(u, v)
            #
        #
    
    return G


class Burrow:
    def __init__(self, map_: NDArray[np.str_]):
        self.map_ = map_.copy()
        self._empty_coords: set[tuple[int, int]] = {
            (i, j) for (i, j), symbol in np.ndenumerate(self.map_) if symbol not in (symbols.wall, symbols.blank)
        }

        # Enumerate all coordinates so we can just refer them by index instead of tuples
        self.coords_ordered = sorted(self._empty_coords)
        self._coords_inv = {c: i for i, c in enumerate(self.coords_ordered)}
        for coord in self._empty_coords:
            self.map_[*coord] = symbols.empty
        
        # Coord for the hallway and rooms
        _hallway_coords = {(i, j) for i, j in self.coords_ordered if i == self.coords_ordered[0][0]}
        _room_coords = self._empty_coords - _hallway_coords
        # The forbidden positions just outside the rooms
        self.entrance_pos = {self._coords_inv[c] for c in _get_neighbor_coords(*_room_coords) & _hallway_coords}

        # The positions where amphipods may stop in the hallway and rooms
        self.hallway_nodes = {self._coords_inv[c] for c in _hallway_coords} - self.entrance_pos
        self.room_nodes = {self._coords_inv[c] for c in _room_coords}
        
        _home_rooms: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for i, j in _room_coords:
            _home_rooms[j].append((i, j))
        
        # Map each amphipod symbol to their home room positions, most southern position first
        self.home_rooms: dict[amphtype, list[int]] = dict()
        for amph_sym, (_, coords) in zip(amph_symbols_ordered, sorted(_home_rooms.items()), strict=True):
            self.home_rooms[amph_sym] = [self._coords_inv[c] for c in sorted(coords, reverse=True)]
        
        # Pre-compute all shortest paths and their lengths
        G = build_graph(*self.coords_ordered, mapper=self._coords_inv)
        self.paths: dict[int, dict[int, set[int]]] = dict()
        self.dists: dict[int, dict[int, int]] = dict()

        for u, d in nx.all_pairs_all_shortest_paths(G):
            assert all(len(p) == 1 for p in d.values())  # check paths are unique
            
            # Store the nodes visited, excluding the starting node
            unpacked = ((v, paths[0]) for v, paths in d.items())
            visited = {v: set(nodes[1:]) for v, nodes in unpacked if len(nodes) > 1}
            self.paths[u] = visited
            self.dists[u] = {v: len(nodes) for v, nodes in visited.items()}
        #

    def display_state(self, state: keytype|None=None) -> None:
        """Helper method for displaying a state"""
        m = self.map_.copy()
        if state:
            for amph in state:
                i, j = self.coords_ordered[amph.pos]
                m[i, j] = amph.kind
            #
        s = "\n".join(["".join(row) for row in m])
        print(s, end="\n\n")

    def state_from_ascii(self, map_: NDArray[np.str_]) -> keytype:
        """Determines a state from an ASCII map"""
        res_list = []
        for (i, j), char in np.ndenumerate(map_):
            sym = char.item()
            if is_amph(sym):
                pos = self._coords_inv[(i, j)]
                res_list.append(Amphipod(pos=pos, kind=sym))
            #
        res_list.sort(key=lambda a: (a.kind, a.pos))
        return tuple(res_list)

    @cache
    def _moves_for_single(self, a: Amphipod, blocked: frozenset[int]) -> list[tuple[int, Amphipod]]:
        res = []
        for target in sorted(self.hallway_nodes):
            path = self.paths[a.pos][target]
            if path.isdisjoint(blocked):
                res.append((movement_costs[a.kind]*len(path), self._move_amphipod(a, target)))
            #
        
        return res

    def _move_amphipod(self, a: Amphipod, to_: int) -> Amphipod:
        """Moves an amphipod. Updates its position and its flags for having moved to hallway/home room"""
        assert not a.moved_to_room
        if to_ in self.hallway_nodes:
            assert not a.moved_to_hallway
            return replace(a, pos=to_, moved_to_hallway=True)
        else:
            assert to_ in self.home_rooms[a.kind]
            return replace(a, pos=to_, moved_to_room=True)
        
    def move_home(self, *amphipods: Amphipod) -> tuple[int, int, Amphipod]|None:
        """Attempts to move an amphipod home.
        The reasoning is, moving an amphipod to its final location can never negatively affect any other
        changes done, because no other paths can go through the node to which we move an amphipod.
        Prioritizing moving amphipods home will therefore keep the search space smaller.
        Returns index, cost, updated_amphipod, or None if none can move home"""

        groups = itertools.groupby(amphipods, operator.attrgetter("kind"))
        blocked = frozenset((a.pos for a in amphipods))
        
        cost = 0
        ind = -1

        for kind, group_ in groups:
            group = list(group_)
            home_coords = self.home_rooms[kind]
            target = home_coords[-len(group)]
            for a in group:
                ind += 1
                nodes = self.paths[a.pos][target]
                if nodes.isdisjoint(blocked):
                    cost = len(nodes)*movement_costs[a.kind]
                    updated = self._move_amphipod(a, target)
                    return ind, cost, updated
                #
            #
        
        return None

    def neighbor_states(self, state: keytype) -> Iterator[tuple[int, keytype]]:
        free_inds, free_amphs = zip(*((i, a) for i, a in enumerate(state) if not a.moved_to_room))
        temp = list(state)

        _attempt_move_home = self.move_home(*free_amphs)
        if _attempt_move_home is not None:
            ind, cost, updated = _attempt_move_home
            yield cost, tuple(a if i != free_inds[ind] else updated for i, a in enumerate(state))
            return
        
        blocked = frozenset((a.pos for a in state))

        for ind, a in enumerate(state):
            if a.moved_to_hallway or a.moved_to_room:
                continue
            original = temp[ind]
            for cost, moved in self._moves_for_single(state[ind], blocked):
                temp[ind] = moved
                yield cost, tuple(temp)
            temp[ind] = original

    def _lower_bound(self, *amphipods: Amphipod) -> int:
        groups = itertools.groupby(amphipods, operator.attrgetter("kind"))
        res = 0
        for kind, group in groups:
            cost = movement_costs[kind]
            for a, target in zip(group, reversed(self.home_rooms[kind])):
                res += self.dists[a.pos][target]*cost
            #
        return res

    def heuristic(self, state: keytype) -> int:
        res = self._lower_bound(*(a for a in state if a.pos not in self.home_rooms[a.kind]))
        return res
        

def a_star(burrow: Burrow, initial_state: keytype, maxiter=-1) -> int:
    d_g = {initial_state: 0}

    f0 = burrow.heuristic(initial_state)
    queue: list[tuple[int, keytype]] = []
    
    # maps states to shortest known path to it
    d_g = {initial_state: 0}  # Shortest known path
    d_f = {initial_state: f0}  # Lower bound from state
    heappush(queue, (f0, initial_state))

    nits = 0
    maxdist = 0
    max_home = 0
    min_h = float('inf')

    while queue:
        if maxiter != -1 and nits >= maxiter:
            break
        nits += 1

        if nits % 10_000 == 0:
            print(f"{nits=}, {len(queue)=}, {maxdist=}, {max_home=}, {min_h}")

        h0, current = heappop(queue)
        maxdist = max(maxdist, h0)
        n_home = sum(a.moved_to_room for a in current)
        max_home = max(max_home, n_home)
        done = n_home == len(current)
        if done:
            dist = d_g[current]
            return dist

        for uv, neighbor in burrow.neighbor_states(current):
            g_tentative = d_g[current] + uv
            
            improved = g_tentative < d_g.get(neighbor, float("inf"))
            if improved:
                d_g[neighbor] = g_tentative
                
                h = burrow.heuristic(neighbor)
                min_h = min(min_h, h)
                f = g_tentative + h
                d_f[neighbor] = f
                heappush(queue, (f, neighbor))
            #
        #
    raise RuntimeError("No path found")


def solve(data: str) -> tuple[int|str, ...]:
    map_ = parse(data)
    burrow = Burrow(map_)

    initial_state = burrow.state_from_ascii(map_)
    burrow.display_state(initial_state)
    star1 = a_star(burrow, initial_state)
    print(f"Solution to part 1: {star1}")

    _more = parse(extra_snippet)
    pad = (map_.shape[1] - _more.shape[1]) // 2
    padded = np.pad(_more, ((0, 0), (pad, pad)), constant_values=symbols.blank)
    large_map = np.vstack((map_[:3], padded, map_[3:]))

    large_burrow = Burrow(large_map)
    start_2 = large_burrow.state_from_ascii(large_map)

    star2 = a_star(large_burrow, start_2)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
