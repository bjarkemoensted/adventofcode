#    •⸳.ꞏ`   +`  ꞏ   *  ⸳`.ꞏ  `         .`•.⸳⸳+ ꞏ* .  ⸳   ⸳ꞏ   *+  •`ꞏ.   ꞏ• +⸳ 
#  +ꞏ⸳`  •  ꞏ * . ` .⸳ Experimental Emergency Teleportation *`   ⸳ ꞏ   `*•  ⸳ ..
#  ⸳`. + ⸳ꞏ  . + `*⸳   https://adventofcode.com/2018/day/23 ꞏ+       ⸳ `* • .` ꞏ
# +ꞏ• . `  *.`   .  ⸳`ꞏ*ꞏ`+ .  ⸳ .*  ⸳ꞏ` .ꞏ+⸳• `   ⸳ ` •.ꞏ•`       ⸳  ꞏ⸳.`*.ꞏ ꞏ⸳

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import re


@dataclass
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

    def signal_edge_dist(self, from_: tuple[int, ...]|None=None) -> tuple[int, int]:
        """Returns a tuple denoting the distnaces from the specified point (default: origo) where the signal
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


def solve(data: str):
    bots = parse(data)
    
    strongest = max(bots, key=lambda bot: bot.r)
    star1 = sum(strongest.in_range(bot) for bot in bots)
    print(f"Solution to part 1: {star1}")

    star2 = closest_point_with_max_signals(bots)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 23
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
