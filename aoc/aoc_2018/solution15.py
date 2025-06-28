# .ꞏ` *⸳   ꞏ.`+  .⸳*ꞏ⸳    ⸳ .    + ` ⸳.* `. `⸳* `* ꞏ.`  . `  +  ⸳`ꞏ*⸳  .  •  . *
# ⸳ꞏ ⸳ ` ꞏ•*ꞏ+.⸳       ꞏ ` *ꞏ +⸳ Beverage Bandits  `  ⸳.ꞏ+   ⸳  ` . •  ⸳•   .*⸳ꞏ
# *⸳ ` . *⸳     ⸳  .+ꞏ https://adventofcode.com/2018/day/15  `⸳     ꞏ *    ꞏ⸳•`.
# ꞏ`⸳     . *  `      * ⸳ ꞏ` .⸳ꞏ*⸳  ⸳ꞏ   ` .⸳*ꞏ⸳ .  •      .⸳ ꞏ⸳   ` *` . `⸳*.•⸳

from __future__ import annotations
from collections import defaultdict
from functools import cache, partial
import networkx as nx
import numpy as np
import time
from typing import Any, Callable, Iterable, Iterator, TypeAlias
from pprint import pprint

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


@cache
def _manhatten_dist(a: coord, b: coord) -> int:
    if a > b:
        return _manhatten_dist(b, a)
    
    res = sum(abs(c1-c2) for c1, c2 in zip(a, b, strict=True))
    return res


_directions: list[coord] = [(1, 0), (0, 1), (-1, 0), (0, -1)]


@cache
def _get_neighbor_sites(coord: coord) -> list[coord]:
    i, j = coord
    res = [(i+di, j+dj) for di, dj in _directions]
    return res


def make_graph_from_ascii_map(map_: np.typing.NDArray[np.str_]) -> nx.Graph:
    res = nx.Graph()
    nodes = {(i, j) for i, j in np.ndindex(map_.shape) if map_[i, j] != _wall_char}
    res.add_nodes_from(nodes)
    res.add_edges_from((u, v) for u in nodes for v in _get_neighbor_sites(u) if v in nodes)
    
    return res


class GameOverException(Exception):
    pass


class Unit: 
    def __init__(self, pos: coord, unit_type: str, cavern: Cavern, hit_points: int=200, attack_power: int=3):
        self.pos = pos
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
        dist = _manhatten_dist(self.pos, other.pos)
        return dist == 1
    
    def _move(self, target: coord):
        if _manhatten_dist(self.pos, target) != 1:
            raise ValueError
        self.pos = target
    
    def _attempt_attack(self) -> bool:
        """Attempts to choose an adjacent enemy and attack them.
        Returns a boolean, indicating whether an attack was made"""
        candidates = [e for e in self.enemies if self._in_range(e)]

        if not candidates:
            return False
        
        target = min(candidates, key=lambda c: (c.hit_points, c.pos))
        assert all(c.hit_points >= target.hit_points for c in candidates)
        self.attack(target)
        return True
    
    def _sites_in_range_of_enemies(self) -> set[coord]:
        res = set().union(*map(set, (self.cavern.G.neighbors(e.pos) for e in self.enemies)))
        return res
    
    def choose_step_towards(self, targets: Iterable[coord]) -> coord|None:
        nearest = self.cavern.determine_nearest(a=self.pos, targets = targets)
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
        
        a = time.time()
        self._free_path_dists = dict(nx.all_pairs_shortest_path_length(self.G))
        self._free_path_nodes = dict(nx.all_pairs_shortest_path(self.G))  # can be cast as sets immediately?
        self.centralities = dict(nx.current_flow_betweenness_centrality(self.G))
        self._high_centrality = sorted(self.centralities.values())[int(len(self.centralities)*0.90)]
        self._heuristic = lambda u, v: self._free_path_dists[u][v]

        self._crits, self._clusters = determine_critical_pints_and_clusters(self.G)
        self._node2cluster = {node: i for i, c in enumerate(self._clusters) for node in c}
        self._compact = collapse_graph(self.G, self._node2cluster)
        
        self._time_spent = 0.0
        
        self._cache: dict[tuple, int|None] = dict()
        self._lower_bound_cache = dict()
        
        self._subgraph_astar_cache: dict[tuple, Callable] = dict()
        self._misc = defaultdict(int)
        self._n = 0
        self._runtime_cent = []
    
    def _snip(self, blocked: Iterable[coord]) -> Callable:
        blocked_set = set(blocked)
        weight_fun = lambda u, v, _: None if v in blocked_set else 1
        return weight_fun

    def lower_bound(self, u: coord, v: coord, blocked: tuple[coord]) -> int|None:
        uc, vc = (self._node2cluster.get(n, n) for n in (u, v))
        blocked_non_cluster = {b for b in blocked if b not in self._node2cluster}
        key = (uc, vc, tuple(sorted(blocked_non_cluster)))
        try:
            return self._lower_bound_cache[key]
        except KeyError:
            pass
        
        # TODO might be faster to cache all pairs hsortest for each distinct blocked signature?
        weight = self._snip(blocked_non_cluster)
        try:
            res = nx.shortest_path_length(G=self._compact, source=uc, target=vc, weight=weight)
        except nx.exception.NetworkXNoPath:
            res = None
        
        self._lower_bound_cache[key] = res

        return res


    def _get_astar(self, blocked: tuple[coord]) -> Callable:
        try:
            res = self._subgraph_astar_cache[blocked]
        except:
            weight = self._snip(blocked)
            heuristic = heuristic=self._heuristic
            res = partial(nx.astar_path, G=self.G, heuristic=heuristic, weight=weight)
            #res = partial(nx.shortest_path, G=self.G, weight=weight)

            self._subgraph_astar_cache[blocked] = res
        return res
    
    def _compute_dist(self, u: coord, v: coord, blocked: tuple[coord, ...], cutoff=None) -> int|None:
        ideal_path = self._free_path_nodes[u][v][1:]
        path_is_blocked = any(node in blocked for node in ideal_path)
        
        if v in blocked:
            self._misc["dumb"] += 1
            return None
        
        noway = cutoff is not None and cutoff < len(ideal_path)
        if noway:
            return None

        if not path_is_blocked:
            self._misc["simplified"] += 1
            return len(ideal_path)
        
        astar = self._get_astar(blocked)
        try:
            if astar.func is nx.shortest_path:
                p = astar(source=u, target=v)
            else:
                p = astar(source=u, target=v, cutoff=cutoff)
            
            res = len(p) - 1
            for i, (up, vp) in enumerate(p):
                self._cache[(up, vp, blocked)] = i
            
        except nx.exception.NetworkXNoPath:
            res = None
        
        
        return res
    
    def compute_dist(self, u: coord, v: coord, blocked: Iterable[coord]|None=None, cutoff=None) -> int|None:
        blocked = tuple(sorted(blocked)) if blocked is not None else ()
        if v in blocked:
            return None
        key = (u, v, blocked)

        lower_bound = self.lower_bound(u, v, blocked)
        if lower_bound is None or (cutoff is not None and lower_bound > cutoff):
            self._misc["killed w lower bound"] += 1
            return None

        try:
            res = self._cache[key]
            self._misc["lookup_success"] += 1
        except KeyError:
            res = self._compute_dist(u, v, blocked, cutoff=cutoff)
            if cutoff is None:
                self._cache[key] = res
        
        # If we found the shortest path via lookup (in paths with no bound on length), check cutoff
        if cutoff is not None:
            if res is not None and res > cutoff:
                res = None
            #
        return res
    
    def __call__(self, u: coord, v: coord, blocked: Iterable[coord]|None=None, cutoff=None) -> int|None:
        _now = time.time()
        self._n += 1
        res = self.compute_dist(u=u, v=v, blocked=blocked, cutoff=cutoff)
        
        
        dt = time.time() - _now
        self._time_spent += dt

        cent = 0
        if blocked:
            obstructed = [node for node in blocked if node in self._free_path_nodes[u][v][1:]]
            if obstructed:
                cent = max(self.centralities[node] for node in obstructed)
        
        # !!!
        len_ideal = len(self._free_path_nodes[u][v][1:])
        import string
        cinds = [string.ascii_lowercase[self._node2cluster.get(n, -1)] for n in (u, v)]
        # !!!!!

        
        lower_bound = self.lower_bound(u, v, blocked)
        #print(cinds, lower_bound, len_ideal, res, f"{cutoff=}", any(n in self._crits for n in self._free_path_nodes[u][v][1:]), dt*1_000_000, blocked)
        #print(f"{lower_bound=}, {res=}")
        #if res is not None and lower_bound is not None:
        #    assert res >= lower_bound
        #print()
        
        self._runtime_cent.append((cent, dt))
        return res



class Cavern:
    def __init__(self, map_: np.typing.NDArray[np.str_]):
        """Sets up a cavern representing the battle map"""
        self.G = make_graph_from_ascii_map(map_)
        self.distcom = DistanceComputer(self.G)
        
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
        self._ugh = 0.0
    
    def reset(self, elf_kwargs: dict|None=None, goblin_kwargs: dict|None=None) -> None:
        self._units = []
        for pos, char in self._initial_unit_positions.items():
            kws = elf_kwargs if char == _elf_char else goblin_kwargs
            if kws is None:
                kws = dict()
            self._units.append(Unit(pos, char, self, **kws))

    def determine_nearest(self, a: coord, targets: Iterable[coord]) -> list[tuple[coord, coord]]:
        """Takes a starting point and an iterable of target points.
        Returns a list of tuples of (target, first_step), with each element representing
        a target site that's tied for closes, and a legal first step on a shortest path there."""
        
        _now = time.time()
        res: list[tuple[coord, coord]] = []
        cutoff = None

        blocked = tuple(sorted(self.occupied_sites))
        #for b in sorted(targets, key = lambda v: self.heuristic_dist(a, v)):
        for b in targets:
            
            dist = self.distcom(u=a, v=b, blocked = blocked, cutoff=cutoff)
            if dist is None:
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
                if u in blocked:
                    continue
                
                dist_post_step = self.distcom(u=u, v=b, blocked=blocked, cutoff=threshold)
                if dist_post_step is None or dist_post_step != dist - 1:
                    continue
                    
                res.append((b, u))
                continue
                
        self._ugh += (time.time() - _now)
        
        return res
    
    
    @property
    def units(self) -> list[Unit]:
        res = [c for c in self._units if c.alive]
        return res
    
    @property
    def occupied_sites(self) -> set[coord]:
        return {c.pos for c in self.units}
    
    def as_string(self, include_units=False, subs:dict[coord, str]|Iterable[coord]|None=None):
        """Represents the cavern as an ASCII map, similarly to on the web page"""
        m = self._mapchars.copy()
        for c in self.units:
            m[*c.pos] = c.char
        
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
    
    def _remove_dead(self):
        """Removes dead units"""
        
        # Go from last to first element so removing elements doesn't mess with the iteration
        for i in range(len(self._units)-1, -1, -1):
            if not self._units[i].alive:
                del self._units[i]
            #
        #
    
    def tick(self):
        order = sorted(range(len(self._units)), key=lambda i: self._units[i].pos)
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


def determine_attack_power_to_keep_elves_alive(cavern: Cavern, verbose = False) -> int:
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
        low = high
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


def identify_choke_points(G: nx.Graph, n: int):
    G = G.copy()
    res = []
    fun = partial(nx.approximate_current_flow_betweenness_centrality, kmax=30)
    
    fun = partial(nx.current_flow_betweenness_centrality)
    while len(res) < n:
        node, cent = max(fun(G).items(), key=lambda t: t[1])
        res.append(node)
        break
    
    return res


def determine_critical_pints_and_clusters(G: nx.G):
    cut_percentile = 95
    centralities = dict(nx.current_flow_betweenness_centrality(G))
    cent_arr = np.array(list(centralities.values()))
    cent_thres = np.percentile(cent_arr, q=cut_percentile)

    critical_points = {node for node in G.nodes() if centralities[node] >= cent_thres}
    clustered = (node for node, deg in nx.degree(G) if deg == 4 and node not in critical_points)
    sg = nx.subgraph(G, clustered)

    cluster_minsize = 10

    # Explan clusters with non-critical points from their periphery
    clusters = [set(nodes) - critical_points for nodes in nx.connected_components(sg)]
    shells = [c for c in clusters]
    for _ in range(5):
        for i, nodesset in enumerate(shells):
            shell = {nb for n in nodesset for nb in G.neighbors(n) if nb not in critical_points} - nodesset
            assert len(shell & critical_points) == 0
            clusters[i] |= shell
            shells[i] = shell
    
    # Join overlapping clusters
    while True:
        n_pre = len(clusters)
        for i in range(len(clusters)-1, -1, -1):
            for j in range(i-1, -1, -1):
                if clusters[i] & clusters[j]:
                    clusters[j] |= clusters.pop(i)
                    break
                #
        if len(clusters) == n_pre:
            break
        #
    
    clusters = [c for c in clusters if len(c) >= cluster_minsize]
    clusters.sort(key=len, reverse=True)
    print("!!!", len(clusters))
    
    return critical_points, clusters


def collapse_graph(G: nx.Graph, node_cluster_map: dict) -> nx.Graph:
    res = nx.Graph()
    for edge in G.edges():
        u, v = (node_cluster_map.get(n, n) for n in edge)
        if u == v:
            continue
        res.add_edge(u, v)
    
    return res


def solve(data: str):
    map_ = parse(data)
    cavern = Cavern(map_)
    
    print(cavern.distcom._compact)
    
    
    crits, clusters = determine_critical_pints_and_clusters(cavern.G)

    
    
    subs = {node: lab for lab, c in zip("ABCDEFGH", clusters) for node in c}
    subs.update({n: " " for n in crits})

    print(cavern.as_string(subs=subs))
    
    
    
    star1 = cavern.play_game()
    print(f"Solution to part 1: {star1}")
    

    star2 = determine_attack_power_to_keep_elves_alive(cavern=cavern, verbose=len(data) > 100)
    print(f"Solution to part 2: {star2}")
    
    print("forbidden nodes:", len(cavern._blocked_registry), len(set(cavern._blocked_registry)))

    desc = lambda n: f"{n}/{cavern.distcom._n} ({100*n/cavern.distcom._n:.1f}%)"
    
    print(f"*** TIME SPENT ON DISTS: {cavern.distcom._time_spent:.5f}***")
    for k, v in cavern.distcom._misc.items():
        print(f"{k}:", desc(v))
    
    cents, dts = zip(*cavern.distcom._runtime_cent)
    from scipy.stats import pearsonr
    print(pearsonr(cents, dts))


    print("**********************")
    return star1, star2


def main():
    year, day = 2018, 15
    #from aoc.utils.data import check_examples
    #check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    #raw = test2  # !!!
    a, b = solve(raw)
    
    if (a, b) == (None, None):
        return
    if len(raw) < 100:
        assert (a, b) == (27730, 4988)
        print("PHEW")
    elif len(raw) > 1000:
        assert (a, b) == (269430, 55160)
        print("PHEEEEEEW")


if __name__ == '__main__':
    main()
