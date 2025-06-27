# .ꞏ` *⸳   ꞏ.`+  .⸳*ꞏ⸳    ⸳ .    + ` ⸳.* `. `⸳* `* ꞏ.`  . `  +  ⸳`ꞏ*⸳  .  •  . *
# ⸳ꞏ ⸳ ` ꞏ•*ꞏ+.⸳       ꞏ ` *ꞏ +⸳ Beverage Bandits  `  ⸳.ꞏ+   ⸳  ` . •  ⸳•   .*⸳ꞏ
# *⸳ ` . *⸳     ⸳  .+ꞏ https://adventofcode.com/2018/day/15  `⸳     ꞏ *    ꞏ⸳•`.
# ꞏ`⸳     . *  `      * ⸳ ꞏ` .⸳ꞏ*⸳  ⸳ꞏ   ` .⸳*ꞏ⸳ .  •      .⸳ ꞏ⸳   ` *` . `⸳*.•⸳

from __future__ import annotations
from functools import cache, partial
import networkx as nx
import numpy as np
from typing import Any, Callable, Iterable, Iterator, TypeAlias

coordtype: TypeAlias = tuple[int, int]

_goblin_char = "G"
_elf_char = "E"
_open_char = "."
_wall_char = "#"

unit_symbols = (_goblin_char, _elf_char)

test = """#########
#G..G..G#
#.......#
#.......#
#G..E..G#
#.......#
#.......#
#G..G..G#
#########"""


test2 = """#######
#.G...#
#...EG#
#.#.#G#
#..G#E#
#.....#
#######"""

test3 = """####### 
#G..#E#
#E#E.E#
#G.##.#
#...#E#
#...E.#
#######"""

test4 = """####### 
#E..EG#
#.#G.E#
#E.##E#
#G..#.#
#..E#.#
#######"""

test5 = """####### 
#E.G#.#
#.#G..#
#G.#.G#
#G..#.#
#...E.#
#######"""

test6 = """####### 
#.E...#
#.#..G#
#.###.#
#E#G#G#
#...#G#
#######"""

test7 = """#########
#G......#
#.E.#...#
#..##..G#
#...##..#
#...#...#
#.G...G.#
#.....G.#
#########"""


def parse(s) -> np.typing.NDArray[np.str_]:
    """Parse input into array of chars"""
    chars = [[char for char in line.strip()] for line in s.splitlines()]
    cols = len(chars[0])
    assert all(len(row) == cols for row in chars)
    
    res = np.array(chars, dtype='<U1')
    return res


@cache
def _manhatten_dist(a: coordtype, b: coordtype) -> int:
    if a > b:
        return _manhatten_dist(b, a)
    
    res = sum(abs(c1-c2) for c1, c2 in zip(a, b, strict=True))
    return res


_directions: list[coordtype] = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def _get_neighbor_sites(coord: coordtype) -> list[coordtype]:
    i, j = coord
    res = [(i+di, j+dj) for di, dj in _directions]
    return res


def make_graph_from_ascii_map(map_: np.typing.NDArray[np.str_]) -> nx.Graph:
    res = nx.Graph()
    nodes = {(i, j) for i, j in np.ndindex(map_.shape) if map_[i, j] != _wall_char}
    res.add_nodes_from(nodes)
    res.add_edges_from((u, v) for u in nodes for v in _get_neighbor_sites(u) if v in nodes)
    
    return res


def _make_edge_hider(forbidden_nodes: Iterable[coordtype]) -> Callable[[coordtype, coordtype, dict], int|None]:
    """This generates and returns a weighting function which can be used in conjunction with the A* algorithm to
    determine shortest paths in a known graph where a few nodes are 'off-limits', e.g. blocked.
    It works by taking an iterable of 'forbidden nodes' which are blocked.
    A function is then defined which returns None for edges (u, v) where either node is blocked off. Otherwise,
    a value of 1 is returned. The A* algorithm in networkx allows specifying a custom weighing function which
    uses None to denote 'hidden' edges, so hopefully this is more efficient than having to define new graph
    with only the subset of nodes that aren't blocked at any given time step."""

    youmustnevergohere: set[coordtype] = {u for u in forbidden_nodes}
    def weight(u: coordtype, v: coordtype, _: dict) -> int|None:
        if any(node in youmustnevergohere for node in (u, v)):
            return None
        else:
            return 1
        #
    
    return weight


class GameOverException(Exception):
    pass


class Unit: 
    def __init__(self, coord: coordtype, unit_type: str, cavern: Cavern, hit_points: int=200, attack_power: int=3):
        self.coord = coord
        if unit_type not in unit_symbols:
            raise ValueError("Invalid unit char")
        
        self.attack_power = attack_power
        self.char = unit_type
        self.cavern = cavern
        self.hit_points = hit_points
        self.alive = True
    #
    
    @property
    def enemies(self):
        return [c for c in self.cavern.units if c.char != self.char]
    
    def receive_damage(self, n_damage: int):
        self.hit_points -= n_damage
        if self.hit_points <= 0:
            self.alive = False
    
    def attack(self, other: Unit):
        assert self._in_range(other)
        other.receive_damage(self.attack_power)
    
    def _in_range(self, other: Unit) -> bool:
        dist = _manhatten_dist(self.coord, other.coord)
        return dist == 1
    
    def _move(self, target: coordtype):
        if _manhatten_dist(self.coord, target) != 1:
            raise ValueError
        self.coord = target
    
    def _attempt_attack(self) -> bool:
        """Attempts to choose an adjacent enemy and attack them.
        Returns a boolean, indicating whether an attack was made"""
        candidates = [e for e in self.enemies if self._in_range(e)]

        if not candidates:
            return False
        
        target = min(candidates, key=lambda c: (c.hit_points, c.coord))
        assert all(c.hit_points >= target.hit_points for c in candidates)
        self.attack(target)
        return True
    
    def _sites_in_range_of_enemies(self) -> set[coordtype]:
        res = set().union(*map(set, (self.cavern.G.neighbors(e.coord) for e in self.enemies)))
        return res
    
    def choose_step_towards(self, targets: Iterable[coordtype]) -> coordtype|None:
        nearest = self.cavern.determine_nearest(a=self.coord, targets = targets)
        if not nearest:
            return None
        
        _, first_step = min(nearest)
        return first_step
    
    def tick(self) -> None:
        # if enemies in range, attack and end turn
        #print(f"Ticking... {self.coord}")
        
        assert self.alive
        
        if not self.enemies:
            raise GameOverException
        attacked = self._attempt_attack()
        if attacked:
            return
        
        sites_near_enemies = self._sites_in_range_of_enemies()
        step_to = self.choose_step_towards(targets=sites_near_enemies)
        if step_to is None:
            return
        
        self._move(step_to)
        # can speedup by attacking only if step_to is in sites_near_enemies? !!!
        self._attempt_attack()

    def __repr__(self):
        return f"{self.char}({self.hit_points})"
    #


class DistanceComputer:
    def __init__(self, G: nx.Graph):
        assert not G.is_directed()
        self.G = G


class Cavern:
    def __init__(self, map_: np.typing.NDArray[np.str_]):
        """Sets up a cavern representing the battle map"""
        self.G = make_graph_from_ascii_map(map_)
        self.distcom = DistanceComputer(self.G)  # !!!
        
        # Store the shortest paths between all sites on the map. Can be used as a heuristic distance
        self._cached_shortest_paths = dict(nx.all_pairs_shortest_path_length(self.G))
        
        self._mapchars = np.full(map_.shape, fill_value=_wall_char, dtype='<U1')
        self._mapchars[*zip(*self.G.nodes())] = _open_char
        
        # This holds the units which will make choices, take actions, etc
        self._initial_unit_positions = {(i, j): char for (i, j), char in np.ndenumerate(map_) if char in unit_symbols}
        self._units: list[Unit] = []
        self.reset()
        
        # Stuff for locating bottlenecks
        self._path_clear_counts = []
        self._blocked_registry = []
    
    def reset(self, elf_kwargs: dict|None=None, goblin_kwargs: dict|None=None) -> None:
        self._units = []
        for pos, char in self._initial_unit_positions.items():
            kws = elf_kwargs if char == _elf_char else goblin_kwargs
            if kws is None:
                kws = dict()
            self._units.append(Unit(pos, char, self, **kws))
            
    
    def heuristic_dist(self, u: coordtype, v: coordtype) -> int|float:
        try:
            return self._cached_shortest_paths[u][v]
        except KeyError:
            return float("inf")
        #
    
    def _setup_astar(self, source: coordtype) -> Callable[..., int]:
        """Returns a callable which takes a target location, and other arguments for networkx' A* implementation,
        and returns the shortest distance."""
        
        # Occupied sites (except the source node) are blocked off, so hide edges involving them
        forbidden_nodes = tuple(sorted(n for n in self.occupied_sites if n != source))
        self._blocked_registry.append(forbidden_nodes)
        weight = _make_edge_hider(forbidden_nodes)
        
        # Return an A* method using the cached heuristic and the weighting that blocks off occupied sites
        a_star = partial(nx.astar_path_length, G=self.G, source=source, heuristic=self.heuristic_dist, weight=weight)
        return a_star
    
    def determine_nearest(self, a: coordtype, targets: Iterable[coordtype]) -> list[tuple[coordtype, coordtype]]:
        """Takes a starting point and an iterable of target points.
        Returns a list of tuples of (target, first_step), with each element representing
        a target site that's tied for closes, and a legal first step on a shortest path there."""
        
        res: list[tuple[coordtype, coordtype]] = []
        cutoff = None
        fun = self._setup_astar(source=a)
        
        #for b in sorted(targets, key = lambda v: self.heuristic_dist(a, v)):
        for b in targets:
            if cutoff is not None and self.heuristic_dist(a, b) > cutoff:
                continue
            try:
                dist = fun(target=b, cutoff=cutoff)
                self._path_clear_counts.append(dist == self.heuristic_dist(a, b))  # !!!
            except nx.exception.NetworkXNoPath:
                continue
            
            # Update cutoff and flush result buffer if we beat the current record
            new_record = cutoff is None or dist < cutoff
            if new_record:
                cutoff = dist
                res = []
            
            assert dist == cutoff
            
            # Determine the valid first steps (that reduce dist by 1) along shortests paths from a -> b
            threshold = max(1, dist-1)
            for u in self.G.neighbors(a):
                try:
                    one_step_closer = fun(source=u, target=b, cutoff=threshold) == dist - 1
                    if one_step_closer:
                        # Add target and first step coordinates to results buffer
                        res.append((b, u))
                    #
                except nx.exception.NetworkXNoPath:
                    pass
                #
            #
        
        return res
    
    
    @property
    def units(self) -> list[Unit]:
        res = [c for c in self._units if c.alive]
        return res
    
    @property
    def occupied_sites(self) -> set[coordtype]:
        return {c.coord for c in self.units}
    
    def as_string(self, include_units=False):
        """Represents the cavern as an ASCII map, similarly to on the web page"""
        m = self._mapchars.copy()
        for c in self.units:
            m[*c.coord] = c.char
        
        lines = []
        unit_lines = []
        for i, row in enumerate(m):
            lines.append("".join(row))
            chars = sorted([c for c in self.units if c.coord[0] == i], key = lambda c: c.coord)
            
            unit_lines.append(", ".join([repr(c) for c in chars]))
        
        maxlen = max(map(len, unit_lines))
        unit_lines = [s + " "*(len(s) - maxlen) for s in unit_lines]
        
        sep = " "*5
        if include_units:
            lines = [a + sep + b for a, b in zip(lines, unit_lines, strict=True)]
        
        res = "\n".join(lines)
        return res
    
    def _remove_dead(self):
        """Removes dead units"""
        
        # Go from last to first element so removing elements doesn't mess with the iteration
        for i in range(len(self._units)-1, -1, -1):
            if not self._units[i].alive:
                del self._units[i]
            #
        #
    
    def tick(self):
        order = sorted(range(len(self._units)), key=lambda i: self._units[i].coord)
        for i in order:
            if not self._units[i].alive:
                continue
            self._units[i].tick()
        
        self._remove_dead()
        return True
    
    def play_game(self, display_status: bool=False, n_max: int|None=None, stop_when: Callable|None=None) -> int|None:
        """Plays a game and returns the outcome. stop_when is an optional callable which takes the cavern object,
        and returns a boolean indicating whether the game should stop. If the game terminates in this way,
        None is returned."""
        
        n = 0
        
        if stop_when is None:
            stop_when = lambda _: False
        
        def disp():
            if display_status:
                print(f"After {n} round{'s'*(n != 1)}:")
                print(self.as_string(include_units=True), end="\n\n")
            
        disp()
        still_going = True
        while still_going:
            if stop_when(self):
                return None
            try:
                self.tick()
            except GameOverException:
                break
            n += 1
            disp()
            
            if n_max is not None and n >= n_max:
                return None
            #
        
        outcome = sum(c.hit_points for c in self.units)*n
        return outcome


def determine_attack_power_to_keep_elves_alive(cavern: Cavern, initial_attack=10, verbose = False) -> int:
    n_elves = sum(c == _elf_char for c in cavern._initial_unit_positions.values())
    elves_died = lambda cavern_: sum(unit.char == _elf_char for unit in cavern_.units) < n_elves
    
    low = 0
    high = 1
    outcomes = dict()
    
    def disp():
        if verbose:
            print(f"{low=}, {high=}")
    
    def elves_survive(n: int):
        nonlocal cavern
        cavern.reset(elf_kwargs=dict(attack_power=n))
        outcome = cavern.play_game(stop_when=elves_died)
        if outcome is not None:
            outcomes[n] = outcome
        
        return not elves_died(cavern)
    
    while not elves_survive(high):
        high *= 2
        disp()
        
    while high - low > 1:
        mid = (high + low) // 2
        if elves_survive(mid):
            high = mid
        else:
            low = mid
        disp()
    
    res = outcomes[high]
    return res


def solve(data: str):
    map_ = parse(data)
    
    
    cavern = Cavern(map_)
    
    
    
    star1 = cavern.play_game()
    # 268065 too low
    # 269430
    print(f"Solution to part 1: {star1}")
    #assert star1 == 27730
    

    star2 = determine_attack_power_to_keep_elves_alive(cavern=cavern, verbose=True)
    print(f"Solution to part 2: {star2}")
    
    print(sum(cavern._path_clear_counts), len(cavern._path_clear_counts))
    
    print("forbidden nodes:", len(cavern._blocked_registry), len(set(cavern._blocked_registry)))

    return star1, star2


def main():
    year, day = 2018, 15
    #from aoc.utils.data import check_examples
    #check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    #raw = test2  # !!!
    solve(raw)


if __name__ == '__main__':
    main()
