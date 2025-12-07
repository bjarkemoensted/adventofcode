#  ·*.· +`.`   ··*  •`    . · ·`   . ·   ·`  *·.  `. ·   ·.  `·.. * .·    · .·`•
# .` ·  .··  *· .  ·`· .   ·.* ·. ` Cafeteria    *· ` ·.  `·  . · `• .·  · .·.*`
# ·.` * ·.`.•·  *.  ·  https://adventofcode.com/2025/day/5  ··  * .·   `·.`·+ ·.
# ·*·`.·  . ·  .`·+ .  `··+  ` .  .`·  *·  .` `  ·.·  * ·`·.  `  *·. · ·`.·  *.·

from __future__ import annotations

import typing as t
from dataclasses import dataclass

boundtype: t.TypeAlias = tuple[int, int]


def untangle_bounds(*bounds: boundtype) -> t.Iterator[boundtype]:
    """Takes some bounds (tuples of lower and upper bounds, inclusive).
    Iterates over corresponding independent bounds, which contain
    the same subset of integers, but do not overlap."""
    
    if not bounds:
        return
    
    # Start with the bound with the smallest lower limit
    bounds_ordered = sorted(bounds, reverse=True)
    a, b = bounds_ordered.pop()

    while bounds_ordered:
        ap, bp = bounds_ordered.pop()
        # If the next one overlaps, join into a single interval
        if ap <= b:
            b = max(b, bp)
        else:
            yield a, b
            a, b = ap, bp
        #
    yield a, b


@dataclass
class IDRange:
    """Represents a range of freshness IDs.
    Supports boolean lookup (if id_ in instance) and size (len(instance))"""

    lower: int
    upper: int
    
    def __post_init__(self):
        if not self.upper >= self.lower:
            raise ValueError("Upper bound must be >= lower bound")
        
    def __contains__(self, id_: int) -> bool:
        """Check if an ID is contained in this instance"""
        return self.lower <= id_ <= self.upper

    def __len__(self) -> int:
        """Determines the number of IDs contained in this instance"""
        return self.upper - self.lower + 1
    

def parse(s: str) -> tuple[list[boundtype], list[int]]:
    freshness_str, ids_str = s.split("\n\n")
    
    ids = [int(line) for line in ids_str.splitlines()]
    bounds = []
    for line in freshness_str.splitlines():
        a, b = map(int, line.split("-"))
        bounds.append((a, b))

    return bounds, ids


def solve(data: str) -> tuple[int|str, ...]:
    id_bounds, ids = parse(data)
    id_ranges = [IDRange(lower, upper) for lower, upper in untangle_bounds(*id_bounds)]
    
    # Count up IDs that are present in any of the fresh ID range
    star1 = sum(any(id_ in id_set for id_set in id_ranges) for id_ in ids)
    print(f"Solution to part 1: {star1}")

    # Sum the size of each range
    star2 = sum(map(len, id_ranges))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
