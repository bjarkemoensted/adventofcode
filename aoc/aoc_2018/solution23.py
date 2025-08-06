#    •⸳.ꞏ`   +`  ꞏ   *  ⸳`.ꞏ  `         .`•.⸳⸳+ ꞏ* .  ⸳   ⸳ꞏ   *+  •`ꞏ.   ꞏ• +⸳ 
#  +ꞏ⸳`  •  ꞏ * . ` .⸳ Experimental Emergency Teleportation *`   ⸳ ꞏ   `*•  ⸳ ..
#  ⸳`. + ⸳ꞏ  . + `*⸳   https://adventofcode.com/2018/day/23 ꞏ+       ⸳ `* • .` ꞏ
# +ꞏ• . `  *.`   .  ⸳`ꞏ*ꞏ`+ .  ⸳ .*  ⸳ꞏ` .ꞏ+⸳• `   ⸳ ` •.ꞏ•`       ⸳  ꞏ⸳.`*.ꞏ ꞏ⸳

from __future__ import annotations
from dataclasses import dataclass
import itertools
from heapq import heappop, heappush
import math
import re
from typing import Generic, TypeAlias, TypeVar


coord_type: TypeAlias = tuple[int, ...]


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


@dataclass(frozen=True)
class BoundingBox:
    limits: tuple[tuple[int, int], ...]

    def _split_interval(self, interval: tuple[int, int]):
        """Split an interval in two equal parts, e.g. (1, 10) -> ((1, 5), (6, 10))"""
        low, high = interval
        mid = (low + high) // 2

        a = (low, mid)
        b = (mid+1, high)
        return a, b

    def _dist_to_interval(self, x: int, interval: tuple[int, int]) -> int:
        """The minimum difference between x, and any number in the input interval, both inclusive"""
        a, b = interval
        if x < a:
            return a - x
        elif x > b:
            return x - b
        else:
            return 0

    def splinter(self) -> list[BoundingBox]:
        """Subdivides the box into 8 smaller boxes"""
        splits = (self._split_interval(limit) for limit in self.limits)
        new_limits = itertools.product(*(splits))
        smaller_boxes = [BoundingBox(interval) for interval in new_limits]
        return smaller_boxes
    
    @property
    def points(self) -> list[coord_type]:
        """An ordered list of points within the bounding box (do not call this with large boxes!)"""
        ranges = (range(low, high+1) for low, high in self.limits)
        res = sorted(itertools.product(*ranges))

        return res

    @property
    def volume(self) -> int:
        res = 1
        for a, b in self.limits:
            len_ = b - a + 1
            res *= len_
        
        return res

    def _min_dist_origo(self):
        res = 0
        for limit in self.limits:
            res += min(map(abs, limit))
        return res

    def __lt__(self, other: BoundingBox):
        return self._min_dist_origo() < other._min_dist_origo()

    def bot_in_range(self, bot: Nanobot):
        dist = sum(self._dist_to_interval(x, interval) for x, interval in zip(bot.vector, self.limits, strict=True))
        return dist <= bot.r


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


def _find_bounding_box_size(bots: list[Nanobot]) -> int:
    """Finds the smallest power of 2 such that a box covering xyz ranges (-x, x) contains all bots"""
    coords = itertools.chain(*(bot.vector for bot in bots))
    max_abs = max(map(abs, coords))

    # Find the lowest power of 2 that's at least the max absolute distance
    base_ = math.log2(max_abs)
    res = 2**math.ceil(base_)
    return res


T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """Priority queue for holding the bounding boxes"""
    def __init__(self):
        self._elems: list[tuple[int, T]] = []
    
    def push(self, elem: T, priority: int):
        item = (priority, elem)
        heappush(self._elems, item)
    
    def pop(self) -> tuple[T, int]:
        priority, elem = heappop(self._elems)
        return elem, priority
    
    def __bool__(self):
        return bool(self._elems)
    
    def __len__(self):
        return len(self._elems)


def determine_optimal_point(bots: list[Nanobot], maxiter: int|None=None):
    """Determines the 'best' point, e.g. the point at which the greatest number of nanobots are in range,
    using distance to origo as tiebreaker"""
    
    # Start with a bounding box large enough to contain all bots
    size = _find_bounding_box_size(bots)
    limits = tuple((-size, +size) for _ in range(3))
    initial_box = BoundingBox(limits)

    queue: PriorityQueue[BoundingBox] = PriorityQueue()
    
    def cost(box):
        n_in_range = sum(box.bot_in_range(bot) for bot in bots)
        min_dist = box._min_dist_origo()
        return -n_in_range, min_dist

    def add_to_queue(*boxes: BoundingBox):
        for box in boxes:
            priority = cost(box)
            queue.push(elem=box, priority=priority)

    add_to_queue(initial_box)
    stop_after = float("inf") if maxiter is None else maxiter
    n_its = 0

    best_box: BoundingBox|None = None
    lowest_cost = (float("inf"), float("inf"))

    while queue:
        
        box, priority = queue.pop()
        if priority > lowest_cost:
            coord, _ = zip(*best_box.limits)
            return coord

        print("OMGOMGOMG", priority, box)

        if box.volume == 1:
            if cost(box) < lowest_cost:
                print("HUZZAH!!!", box)
                lowest_cost = cost(box)
                best_box = box
            #
        else:
            add_to_queue(*box.splinter())

        n_its += 1

        if n_its >= stop_after:
            break
        #
    raise RuntimeError
        


def solve(data: str):
    bots = parse(data)
    
    strongest = max(bots, key=lambda bot: bot.r)
    star1 = sum(strongest.in_range(bot) for bot in bots)
    print(f"Solution to part 1: {star1}")

    best_spot = determine_optimal_point(bots)
    
    
    star2 = sum(map(abs, best_spot))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2018, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    #raw = test
    solve(raw)


if __name__ == '__main__':
    main()
