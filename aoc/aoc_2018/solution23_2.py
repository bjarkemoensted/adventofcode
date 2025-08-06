#    •⸳.ꞏ`   +`  ꞏ   *  ⸳`.ꞏ  `         .`•.⸳⸳+ ꞏ* .  ⸳   ⸳ꞏ   *+  •`ꞏ.   ꞏ• +⸳ 
#  +ꞏ⸳`  •  ꞏ * . ` .⸳ Experimental Emergency Teleportation *`   ⸳ ꞏ   `*•  ⸳ ..
#  ⸳`. + ⸳ꞏ  . + `*⸳   https://adventofcode.com/2018/day/23 ꞏ+       ⸳ `* • .` ꞏ
# +ꞏ• . `  *.`   .  ⸳`ꞏ*ꞏ`+ .  ⸳ .*  ⸳ꞏ` .ꞏ+⸳• `   ⸳ ` •.ꞏ•`       ⸳  ꞏ⸳.`*.ꞏ ꞏ⸳

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Iterable, Iterator, TypeAlias


keytype: TypeAlias = tuple["Nanobot", ...]


test = """pos=<10,12,12>, r=2
pos=<12,14,12>, r=2
pos=<16,12,12>, r=4
pos=<14,14,14>, r=6
pos=<50,50,50>, r=200
pos=<10,10,10>, r=5"""


@dataclass(frozen=True)
class Nanobot:
    x: int
    y: int
    z: int
    r: int
    
    def dist(self, other: Nanobot) -> int:
        """Returns Manhatten distance to another nanobot"""
        res = sum(abs(a-b) for a, b in zip(*(bot.vector for bot in (self, other))))
        return res

    def in_range(self, other: Nanobot) -> bool:
        return self.dist(other) <= max(bot.r for bot in (self, other))
    
    def overlap(self, other: Nanobot) -> bool:
        _dist = self.dist(other)
        overlap_width = self.r + other.r - _dist
        return overlap_width >= 0

    def signal_edge_dist(self, from_: tuple[int, ...]|None=None) -> tuple[int, int]:
        """Returns a tuple denoting the distances from the specified point (default: origo) where the signal
        from the bot starts and stops, both inclusive"""
        if from_ is None:
            from_ = tuple(0 for _ in self.vector)
        
        dist_center = sum((abs(a-b) for a, b in zip(self.vector, from_, strict=True)))
        res = (max(0, dist_center - self.r), dist_center + self.r)
        return res

    @property
    def vector(self):
        return (self.x, self.y, self.z)
    
    def __repr__(self):
        return f"pos=<{self.x},{self.y},{self.z}, r={self.r}>"
    #


def parse(s) -> list[Nanobot]:
    res = []
    pattern = re.compile(r"pos=<(?P<x>-?\d+),(?P<y>-?\d+),(?P<z>-?\d+)>, r=(?P<r>\d+)")
    for line in s.splitlines():
        m = re.match(pattern, line)
        if m is None:
            raise RuntimeError
        
        kwds = {k: int(v) for k, v in m.groupdict().items()}
        bot = Nanobot(**kwds)
        res.append(bot)

    return res


def closest_point_with_max_signals(bots: list[Nanobot]) -> int:
    """Determines the closest distnace to the origin at which the maximum possible number of
    nanobots are in range."""
    
    # Keep track of the distances from origin at which the number of in-range bot changes
    increments: dict[int, int] = defaultdict(int)
    signal_edges = [bot.signal_edge_dist() for bot in bots]
    for sig_min_dist, sig_max_dist in signal_edges:
        increments[sig_min_dist] += 1  # add one when we enter the min dist to a new bot
        increments[sig_max_dist+1] -= 1  # subtract one just after we leave the signal range of the bot
    
    # Iterate over all distances to origin where the number of bots in range changes, keeping the max
    steps = sorted((k, v) for k, v in increments.items() if v != 0)
    res = 0
    running = 0
    max_bots_in_range = 0
    for dist, delta_n_bots in steps:
        running += delta_n_bots
        if running > max_bots_in_range:
            max_bots_in_range = running
            res = dist
    
    return res


class PathFinder:
    def __init__(self, bots: list[Nanobot]):
        self.overlaps = {b: set() for b in bots}
        self._cache = dict()

        for i, b1 in enumerate(bots):
            for b2 in bots[i+1:]:
                if b1.overlap(b2):
                    self.overlaps[b1].add(b2)
                    self.overlaps[b2].add(b1)
                #
            #
        
        self._n = 0
        self._best = -1
        self._hits = 0
        self._items = sorted(self.overlaps.items(), key=lambda t: len(t[1]), reverse=True)
        self.bots = [bot for bot, _ in self._items]
        self._inds = {bot: i for i, bot in enumerate(self.bots)}
        self._bots_set = set(self.bots)

    def _get_new_key(self, bot: Nanobot, old_bots: Iterable[keytype]) -> keytype:
        bots_list = list(old_bots) + [bot]
        bots_list.sort(key=self._inds.get)
        res = tuple(bots_list)
        return res

    def valid_next(self, bots: keytype) -> Iterator[tuple[int, Nanobot]]:
        """Take some nanobots. Iterate over valid choices for next addition to overlaps,
        i.e. get the bots which overlap with all input bots.."""

        bs = set(bots)

        for bot, neighborhood in self._items:
            
            if bs.issubset(neighborhood) and len(neighborhood) > len(bs):
                remaining = len(neighborhood) - len(bs)
                yield remaining, bot

    def go(self, allocated: keytype=()) -> tuple[int, list[keytype]]:
        """Return a list of tuples of nanobots that are tied for largest number of bots
        with overlapping signal areas."""

        hit = False
        try:
            res = self._cache[allocated]
            hit = True
            self._hits += 1
        except KeyError:
            pass

        if not hit:

            n_max = len(allocated)
            best = {allocated}

            for n_rem, bot in self.valid_next(allocated):
                if len(allocated) + n_rem < self._best:
                    continue
                key = self._get_new_key(bot, allocated)
                n_deeper, rec = self.go(key)
                if n_deeper > n_max:
                    n_max = n_deeper
                    best = set()
                if n_deeper == n_max:
                    best |= set(rec)
                
            #
            best_list = list(best)
            res = n_max, best_list

        self._cache[allocated] = res
        

        nm, bl = res
        self._n += 1
        self._best = max(nm, self._best)
        if self._n % 10000 == 0:
            print(self._n, self._hits, self._best)

        return res
            
            
        

def solve(data: str):
    bots = parse(data)
    print(len(bots))
    
    strongest = max(bots, key=lambda bot: bot.r)
    star1 = sum(strongest.in_range(bot) for bot in bots)
    print(f"Solution to part 1: {star1}")

    wat = PathFinder(bots)
    n, group = wat.go()
    print(n)

    for ugh in group:
        print(ugh)

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2



def main():
    year, day = 2018, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    #raw = test
    
    # !!!
    a, b = solve(raw)
    #assert a == 319
    #assert b == 129293598



if __name__ == '__main__':
    main()
