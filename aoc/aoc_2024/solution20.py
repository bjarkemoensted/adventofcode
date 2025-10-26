# .··*`  ·  .· + *· `·.  ·+   ·   ` · .·`+ *· ·`. ·  .+  ·`     ·  ·. · +*· ·.`·
# ·`.· `  · ·*      ·+·`• ·  ·`.  Race Condition `*   ·.    ·` ·  . ·  *· `·+ .·
# ·.   ·     · ·  ` *  https://adventofcode.com/2024/day/20  ·   · . ·•`.·*  ··.
# `· .  +·*· .·`   ·.` ·  *·     .  ·`• .·· *   ·   · `· * ·+.·   ·+ .`·  ·`.*·`


from functools import cache
from itertools import product
import networkx as nx
import numpy as np
import warnings


raw = """###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############"""


def parse(s: str):
    #s = raw  # !!!
    res = np.array([list(line) for line in s.splitlines()])
    return res


@cache
def get_offsets(n_steps: int, only_outer=False):
    """Returns a list of tuples of cumulative displacements and sitance possible by taking n steps in
    the NSWE directions. If only_outer is True, returns only displacements reachable by using all steps."""
    
    res = []
    
    for di in range(-n_steps, n_steps+1):
        remaining = n_steps - abs(di)
        for dj in range(-remaining, remaining+1):
            offset = (di, dj)
            dist = sum(map(abs, offset))
            
            if only_outer and dist < n_steps:
                continue
            res.append((offset, dist))
        #
    
    return res


def build_graph(m: np.ndarray, wall: str) -> nx.Graph:
    """Builds the graph, connecting neighboring non-wall coordinates"""
    G = nx.Graph()
    for u in np.ndindex(m.shape):
        if m[*u] == wall:
            continue
        i, j = u
        for (di, dj), _ in get_offsets(n_steps=1, only_outer=True):
            v = (i+di, j+dj)
            in_bounds = all(0 <= c < dim for c, dim in zip(v, m.shape))
            if not in_bounds or m[*v] == wall:
                continue
            G.add_edge(u, v)
        #
    return G


class RaceGraph:
    """Represents the map on which the program can race.
    Nodes are stored as (i, j)-coords."""
    
    start = "S"
    end = "E"
    wall = "#"
    empty = "."
    
    def __init__(self, map_: np.ndarray):
        self.m = map_.copy()
        self.G = build_graph(m=map_, wall=self.wall)
        
        locs = [np.argwhere(self.m == char) for char in (self.start, self.end)]
        assert all(len(crds) == 1 for crds in locs)
        self.start_coord, self.end_coord = (tuple(map(int, crds[0])) for crds in locs)
        
        # Precompute shortest distances from the start to any coord, and from any coord to
        with warnings.catch_warnings():
            # Catch futurewarning that result will be a dict in the future.
            warnings.simplefilter("ignore", FutureWarning)
            self._dists_to_target = dict(nx.single_target_shortest_path_length(G=self.G, target=self.end_coord))
            self._dists_from_source = dict(nx.single_source_shortest_path_length(G=self.G, source=self.start_coord))
        #
    
    def count_cheats(self, n_cheat_steps: int, min_save: int=1):
        """Returns the number of distinct valid cheats (with at most n_cheat_steps steps) which reduce
        the distance by at least the specified value."""

        best_honest = self._dists_to_target[self.start_coord]
        cutoff = best_honest - min_save
        
        res = 0
        
        # Iterate over all possible start and stop points for cheating
        for (i, j), a in self._dists_from_source.items():
            for offset, dist in get_offsets(n_steps=n_cheat_steps):
                di, dj = offset
                cheat_to = (i+di, j+dj)
                
                # Determine the distance when using the cheat
                try:
                    b = self._dists_to_target[cheat_to]
                except KeyError:
                    continue  # Disregard cheat if it takes us to an invalid location
                
                # Include the cheat if it confers a sufficient reduction in the min length
                newdist = a + dist + b
                res += newdist <= cutoff
            #
        return res
    #


def solve(data: str) -> tuple[int|str, int|str]:
    m = parse(data)
    race_track = RaceGraph(m)
    min_save = 100 if len(m) > 20 else 1
    
    star1 = race_track.count_cheats(n_cheat_steps=2, min_save=min_save)
    print(f"Solution to part 1: {star1}")
    
    min_save = 100 if len(m) > 20 else 50
    star2 = race_track.count_cheats(n_cheat_steps=20, min_save=min_save)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 20
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()