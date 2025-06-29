# .ꞏ` *⸳   ꞏ.`+  .⸳*ꞏ⸳    ⸳ .    + ` ⸳.* `. `⸳* `* ꞏ.`  . `  +  ⸳`ꞏ*⸳  .  •  . *
# ⸳ꞏ ⸳ ` ꞏ•*ꞏ+.⸳       ꞏ ` *ꞏ +⸳ Beverage Bandits  `  ⸳.ꞏ+   ⸳  ` . •  ⸳•   .*⸳ꞏ
# *⸳ ` . *⸳     ⸳  .+ꞏ https://adventofcode.com/2018/day/15  `⸳     ꞏ *    ꞏ⸳•`.
# ꞏ`⸳     . *  `      * ⸳ ꞏ` .⸳ꞏ*⸳  ⸳ꞏ   ` .⸳*ꞏ⸳ .  •      .⸳ ꞏ⸳   ` *` . `⸳*.•⸳

from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import functools
import networkx as nx
import numpy as np
from pprint import pprint
import time
from types import MethodType
from typing import Callable, Iterable, TypeAlias


coord: TypeAlias = tuple[int, int]

_goblin_char = "G"
_elf_char = "E"
_open_char = "."
_wall_char = "#"

unit_symbols = (_goblin_char, _elf_char)


def parse(s) -> np.typing.NDArray[np.str_]:
    """Parse input into array of chars"""
    chars = [[char for char in line.strip()] for line in s.splitlines()]
    cols = len(chars[0])
    assert all(len(row) == cols for row in chars)
    
    res = np.array(chars, dtype='<U1')
    return res



@functools.cache
def _get_neighbor_sites(coord: coord) -> list[coord]:
    i, j = coord
    res = [(i+di, j+dj) for di, dj in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
    return res


class GameOverException(Exception):
    """For stopping the game when only one side remains"""
    pass


class Cache:
    """Helper class for caching instance methods.
    Made a custom thing for this to make it simpler to display summary statistics on which
    calls took most time etc, without having to switch profilers on/off and such.
    There's a class method Cache.display_all which prints such info on all instantiated caches."""
    
    _instances: list[Cache] = []

    @classmethod
    def _register(cls, inst):
        cls._instances.append(inst)

    def __init__(self, func):
        """Set up cache for method"""
        self._func_qualname = func.__qualname__
        functools.update_wrapper(self, func)
        self.func = func
        self.cache = dict()
        self.details = {
            "time": 0.0,
            "calls": 0,
            "hits": 0,
            "key_counts": defaultdict(int),
            "key_compute_time": defaultdict(float)
        }

        # Register this cache instance to the class-level registry
        self._register(self)

    def __call__(self, *args, **kwargs):
        _now = time.time()
        key = functools._make_key(args=args, kwds=kwargs, typed=False)

        hit = False
        try:
            res = self.cache[key]
            hit = True
        except KeyError:
            res = self.func(*args, **kwargs)
            self.cache[key] = res
        
        # Update stats
        dt = time.time() - _now
        self.details["key_counts"][key] += 1
        self.details["calls"] += 1
        self.details["hits"] += hit
        self.details["time"] += dt
        if not hit:
            self.details["key_compute_time"][key] += dt
        return res
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return MethodType(self, instance)


    def trimmed(self, top_n: int=10):
        d = deepcopy(self.details)
        for k in ("key_counts", "key_compute_time"):
            d[k] = dict(sorted(d[k].items(), key=lambda t: t[1], reverse=True)[:top_n])
        
        return d

    def display(self, top_n: int=10):
        print(f"*** {self._func_qualname} ***")
        printkeys = ("time", "calls", "hits")
        pprint(dict([(k, v) for k,v in self.details.items() if k in (printkeys)]))
        compute_running: dict[str,float] = defaultdict(float)
        for k, v in self.cache.items():
            compute_running[str(v)] += self.details["key_compute_time"][k]
        
        slow = sorted(compute_running.items(), key=lambda t: t[1], reverse=True)[:top_n]
        print(slow)
    
    @classmethod
    def display_all(cls, top_n: int=10):
        for inst in sorted(cls._instances, key=lambda inst: inst.details["time"], reverse=True):
            inst.display(top_n=top_n)
            print()
        #



class Grid:
    """Handles logistics, inter-sites distances etc"""
    
    def __init__(self, G: nx.Graph):
        self.G = G

    @classmethod
    def from_ascii(cls, map_: np.typing.NDArray[np.str_], **kwargs) -> Grid:
        G = nx.Graph()
        nodes = {(i, j) for i, j in np.ndindex(map_.shape) if map_[i, j] != _wall_char}
        G.add_nodes_from(nodes)
        G.add_edges_from((u, v) for u in nodes for v in _get_neighbor_sites(u) if v in nodes)
        return cls(G, **kwargs)

    def make_weighting_func(self, blocked: Iterable[coord]) -> Callable[..., int|float|None]:
        """Given the currently blocked nodes, makes a 'weight function' which returns None (blocked)
        for any edge going to one of the blocked nodes, and 1 otherwise. This is a trick to avoid
        having to manipulate/copy the entire graph every time something changes."""

        if not isinstance(blocked, set):
            blocked = set(blocked)
        
        def w(u: coord, v: coord, _):
            if v in blocked:
                return None
            return 1
        
        return w

    @Cache
    def dist(self, u: coord, v: coord, blocked) -> int|None:
        """Computes the distance between u and v (None if no path exists)"""
        if v in blocked:
            return None
        
        weight = self.make_weighting_func(blocked)
        try:
            res = nx.shortest_path_length(G=self.G, source=u, target=v, weight=weight)
        except nx.exception.NetworkXNoPath:
            res = None
        return res

    @Cache
    def _get_tied(self, source: coord, targets: tuple[coord], blocked: tuple[coord]) -> None|tuple[int, list[coord]]:
        """Given a sourve and a tuple of target nodes, returns a list of target nodes that are
        tied as being closest to the source, given the currently blocked nodes.
        This method is the one that very easily becomes a major bottleneck."""

        # Weigh edges as usually, except don't block the source node
        weight = self.make_weighting_func(tuple(n for n in blocked if n != source))

        try:
            # Use multisource Dijkstra to get the shortest distance between source and any target
            shortest_dist, _ = nx.multi_source_dijkstra(self.G, sources=targets, target=source, weight=weight)
        except nx.exception.NetworkXNoPath:
            # If that fails, there's no paths
            return None
        
        # No we have the shortest distance (n) do the first n Dijkstra steps and see which nodes are reachable
        _, dists = nx.dijkstra_predecessor_and_distance(self.G, source=source, weight=weight, cutoff=shortest_dist)
        tied = sorted(set(dists.keys()) & set(targets))

        return shortest_dist, tied

    def get_tied_for_closest(self, source: coord, targets: Iterable[coord], blocked: tuple[coord]) -> None|tuple[int, list[coord]]:
        """Returns the shortest distance to any target, and the targets tied at that distance"""
        targets = tuple(sorted(targets))
        res = self._get_tied(source, targets, blocked)
        return res


class Unit:
    """Handles unit logic, deciding actions, giving/taking damage, etc"""
    
    def __init__(self, pos: coord, type_: str, battle: Battle, hit_points: int=200, attack_power: int=3):
        self.pos: coord = pos
        self.type_ = type_
        self.battle = battle
        self.grid: Grid = battle.grid
        self.hit_points = hit_points
        self.attack_power = attack_power

    def __repr__(self):
        return f"{self.type_}({self.hit_points})"

    def attack(self, other: Unit):
        self.is_enemy(other)
        other.hit_points -= self.attack_power

    def neighbor_sites(self) -> set[coord]:
        return set(self.grid.G.neighbors(self.pos))
    
    def is_enemy(self, other: Unit) -> bool:
        return self.type_ != other.type_

    def get_enemies(self) -> list[Unit]:
        return [unit for unit in self.battle.units if self.is_enemy(unit)]

    def choose_enemy_to_attack(self) -> Unit|None:
        enemies = self.get_enemies()
        if not enemies:
            raise GameOverException

        ns = self.neighbor_sites()
        candidates = [enemy for enemy in enemies if enemy.pos in ns]
        if candidates:
            return min(candidates, key=lambda unit: (unit.hit_points, unit.pos))
        else:
            return None

    @property
    def is_alive(self) -> bool:
        return self.hit_points > 0

    def _movement_candidate_sites(self) -> Iterable[coord]:
        """Determine the sites the unit would like to travel to (in range of opponents)"""
        res: set[coord] = set()
        blocked = self.battle.blocked_set
        eligible = (e.neighbor_sites() for e in self.get_enemies())
        res = set().union(*eligible) - blocked
        return res

    def attempt_move(self):
        """Attempts to decide on a site to move towards, and take a step"""
        
        # Abort if there's no sites to go to
        candidates = self._movement_candidate_sites()
        if not candidates:
            return
        
        # Among candidate sites, determine the nearest ones
        blocked_set = self.battle.blocked_set
        blocked = self.battle.blocked
        temp = self.grid.get_tied_for_closest(source=self.pos, targets=candidates, blocked=blocked)
        if not temp:
            return
        
        # Use reading order as tiebreaker
        dist, tied = temp
        target_coord = min(tied)

        # Consider possible first steps (in reading order). Keep the first which is on a shortest path
        first_step_candidates = sorted(self.neighbor_sites() - blocked_set)
        for step in first_step_candidates:
            if self.grid.dist(step, target_coord, blocked) == dist - 1:
                self.pos = step
                return
            #
        raise RuntimeError


    def tick(self):
        opponent = self.choose_enemy_to_attack()
        if opponent:
            self.attack(opponent)
            return
        
        self.attempt_move()

        opponent = self.choose_enemy_to_attack()
        if opponent:
            self.attack(opponent)


class Battle:
    """Handles battle logic, unit turns, end conditions, etc"""
    
    def __init__(self, map_: np.typing.NDArray[np.str_]):
        self.grid: Grid = Grid.from_ascii(map_)

        self._mapchars = np.full(map_.shape, fill_value=_wall_char, dtype='<U1')
        self._mapchars[*zip(*self.grid.G.nodes())] = _open_char

        # This holds the units which will make choices, take actions, etc
        self._initial_unit_positions = {(i, j): char for (i, j), char in np.ndenumerate(map_) if char in unit_symbols}
        self._units: list[Unit] = []
        self.end_condition: None|Callable[[Battle], bool] = None
        self.reset()
    
    @property
    def blocked(self) -> tuple[coord, ...]:
        return tuple(sorted(unit.pos for unit in self.units))

    @property
    def blocked_set(self) -> set[coord]:
        return {unit.pos for unit in self.units}

    def reset(self, elf_kwargs: dict|None=None, goblin_kwargs: dict|None=None) -> None:
        self._units = []
        for pos, char in self._initial_unit_positions.items():
            kws = elf_kwargs if char == _elf_char else goblin_kwargs
            if kws is None:
                kws = dict()
            self._units.append(Unit(pos, char, self, **kws))
    
    @property
    def units(self) -> list[Unit]:
        return [unit for unit in self._units if unit.is_alive]

    def _remove_dead(self):
        """Removes dead units"""
        # Go from last to first element so removing elements doesn't mess with the iteration
        for i in range(len(self._units)-1, -1, -1):
            if not self._units[i].is_alive:
                del self._units[i]
            #
        #

    def tick(self):
        for unit in sorted(self.units, key=lambda u: u.pos):
            if not unit.is_alive:
                continue
            unit.tick()
        self._remove_dead()
    
    def fight(self, display=False, end_condition: Callable[[Battle], bool]|None=None) -> int|None:
        """Fights an elf-goblin battle. end_condition is an optional callable which takes the battle
        instance and returns True if the fight should end.
        Returns the 'outcome' (n_rounds * sum hit points) if game ends."""

        if end_condition is None:
            end_condition = lambda _: False
                
        game_over = False

        n_rounds_fought = 0
        while not game_over:
            try:
                self.tick()
                n_rounds_fought += 1
            except GameOverException:
                game_over = True

            if display:
                print(f"After {n_rounds_fought} round{'s'*(n_rounds_fought != 1)}:")
                print(self.as_string(include_units=True), end="\n\n")
            
            game_over = game_over or end_condition(self)

        outcome = n_rounds_fought*sum(unit.hit_points for unit in self.units) if game_over else None
        return outcome
            

    
    def as_string(self, include_units=True, subs:dict[coord, str]|Iterable[coord]|None=None):
        """Represents the cavern as an ASCII map, similarly to on the web page"""
        m = self._mapchars.copy()
        for c in self.units:
            m[*c.pos] = c.type_
        
        if subs is None:
            subs = dict()
        if not isinstance(subs, dict):
            subs = {pos: 'X' for pos in subs}
        
        lines = []
        unit_lines = []
        for i, row in enumerate(m):
            lines.append("".join([subs.get((i, j), c) for j, c in enumerate(row)]))
            chars = sorted([c for c in self.units if c.pos[0] == i], key = lambda c: c.pos)
            
            unit_lines.append(", ".join([repr(c) for c in chars]))
        
        maxlen = max(map(len, unit_lines))
        unit_lines = [s + " "*(len(s) - maxlen) for s in unit_lines]
        
        sep = " "*5
        if include_units:
            lines = [a + sep + b for a, b in zip(lines, unit_lines, strict=True)]
        
        res = "\n".join(lines)
        return res


def determine_attack_power_to_keep_elves_alive(battle: Battle) -> int:
    """Determines the minimum attack power required before elves will all survies the battle.
    Repeatedly resets the battle and uses binary search to determine the least required attack."""
    
    # Condition for stopping a battle early
    n_elves = sum(c == _elf_char for c in battle._initial_unit_positions.values())
    elves_died = lambda battle_: sum(unit.type_ == _elf_char for unit in battle_.units) < n_elves
    
    low = 0
    high = 16
    outcomes = dict()
    
    # Helper method for determining whether the elves survive at a given attack power
    def elves_survive(n: int) -> bool:
        nonlocal battle, outcomes
        battle.reset(elf_kwargs=dict(attack_power=n))
        # Just record the outcome (if any). We can grab the correct one later
        outcome = battle.fight(end_condition=elves_died)
        if outcome is not None:
            outcomes[n] = outcome
        
        return not elves_died(battle)
    
    # Keep increasing 'high' until we hit a point where the elves make it
    while not elves_survive(high):
        low = high
        high *= 2
    
    # Keep narrowing the range until we find the turning point
    while high - low > 1:
        mid = (high + low) // 2
        if elves_survive(mid):
            high = mid
        else:
            low = mid
    
    res = outcomes[high]
    return res


def solve(data: str):
    map_ = parse(data)
    
    battle = Battle(map_)
    
    star1 = battle.fight(display=False)
    print(f"Solution to part 1: {star1}")
    
    star2 = determine_attack_power_to_keep_elves_alive(battle=battle)
    print(f"Solution to part 2: {star2}")

    #Cache.display_all()  # Display cache statistics to figure out hwere to optimize
    
    return star1, star2


def main():
    year, day = 2018, 15
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
