# ·.`·*`·   •.· · `*   ·  `· . ·*`.·    ·. •· · ` *   +· `  . `· . +· * •· `.· ·
# *··`. .*  ·    ·  `  ` .· •   · *· Amphipod ` *·.+ `·. *·` ·   ·.`    ` ·· .+*
# ·.*`··*   `·  .  ··  https://adventofcode.com/2021/day/23   + ·*`  · `·.+ · `.
# `·.    ··.*     *`  ·. ·  `·*`.   · ·* ·+`   .·.`   `*· ·   ·   · . ·* ·*   ·`

from collections import defaultdict
from dataclasses import dataclass, replace
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


@dataclass(frozen=True, slots=True, order=True)
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
        # TODO use helper function!!!!!
        for v_coord in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
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
        self.dists = dict()

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

    def get_target_state(self, map_: NDArray[np.str_]) -> keytype:
        """Determines the target state, with all "A" amphipods in the leftmost tunnel, etc."""
        m = map_.copy()
        pos_and_amph_chars = (((i, j), char) for (i, j), char in np.ndenumerate(m)if is_amph(char))
        pos, chars = map(list, zip(*pos_and_amph_chars))
        # Sort position and symbols independently, then update ascii and determine corresponding state
        pos.sort(key=lambda tup_: tup_[1])
        chars.sort()
        for (i, j), c in zip(pos, chars):
            m[i, j] = c
        
        return self.state_from_ascii(m)

    def attempt_move_home(self, state: keytype) -> tuple[int, keytype]|None:
        res_list = list(state)

        cost = 0
        modified = False
        for kind, nodes in self.home_rooms.items():
            for node in nodes:
                try:
                    i, occupier = next((i, a) for i, a in enumerate(res_list) if a.pos == node)
                    # If the site is occupied by a correct type amphipos, ensure its moved_to_room is True
                    if occupier.kind == kind:
                        if not occupier.moved_to_room:
                            res_list[i] = replace(occupier, moved_to_room=True)
                            modified = True
                        #
                    else:
                        # This room is currently blocked off by a wrong type amphipod. Proceed to next room
                        break
                except StopIteration:
                    # If the site is free, see if any amphipods have a clear path to the site
                    candidates = ((i, a) for i, a in enumerate(res_list) if a.kind == kind and not a.moved_to_room)
                    occupied = {a.pos for a in res_list}
                    for i, candidate in candidates:
                        path = self.paths[candidate.pos][node]
                        path_is_clear = path.isdisjoint(occupied)
                        if path_is_clear:
                            cost += len(path)*movement_costs[candidate.kind]
                            res_list[i] = replace(candidate, pos=node, moved_to_room=True)
                            modified = True
                            break
                        #
                    #
                #
            #
        
        if modified:
            return cost, tuple(res_list)
        else:
            return None
        #

    def get_possible_moves(self, amph: Amphipod, *others: Amphipod) -> Iterator[tuple[int, Amphipod]]:
        # TODO maybe look for obvious choices like moving unobstructed amphs to their destination!!!
        unit_cost = movement_costs[amph.kind]
        pos = amph.pos
        occupied = {a.pos: a.kind for a in others}
        if not amph.moved_to_hallway:
            assert pos in self.room_nodes

            for node_hw in self.hallway_nodes:
                path = self.paths[pos][node_hw]
                if not path.isdisjoint(occupied):
                    continue

                updated = replace(amph, pos=node_hw, moved_to_hallway=True)
                yield unit_cost*len(path), updated
            #

    def neighbor_states(self, state: keytype) -> Iterator[tuple[int, keytype]]:
        # If we can move any amphipods to their home locations, just do that.
        moved_home_state = self.attempt_move_home(state)
        if moved_home_state:
            yield moved_home_state
            return
        
        temp = list(state)
        for i, amph in enumerate(state):
            others = [a for j, a in enumerate(state) if j != i]
            for cost, new_amph in self.get_possible_moves(amph, *others):
                new_state = tuple(new_amph if j == i else oldamp for j, oldamp in enumerate(temp))
                yield cost, new_state
                #
            #
        #

    def heuristic(self, state: keytype) -> int:
        res = 0
        used = set()
        for a in state:
            if a.pos in self.home_rooms[a.kind]:
                continue
            closest = None
            record = -1
            for home_pos in self.home_rooms[a.kind]:
                if home_pos in used:
                    continue
                dist = self.dists[a.pos][home_pos]
                if record == -1 or dist < record:
                    record = dist
                    closest = home_pos
                #
            
            assert closest is not None
            used.add(closest)
            res += record*movement_costs[a.kind]

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
