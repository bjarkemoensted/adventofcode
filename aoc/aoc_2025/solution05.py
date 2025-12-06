#  ·*.· +`.`   ··*  •`    . · ·`   . ·   ·`  *·.  `. ·   ·.  `·.. * .·    · .·`•
# .` ·  .··  *· .  ·`· .   ·.* ·. ` Cafeteria    *· ` ·.  `·  . · `• .·  · .·.*`
# ·.` * ·.`.•·  *.  ·  https://adventofcode.com/2025/day/5  ··  * .·   `·.`·+ ·.
# ·*·`.·  . ·  .`·+ .  `··+  ` .  .`·  *·  .` `  ·.·  * ·`·.  `  *·. · ·`.·  *.·

from __future__ import annotations

import typing as t
from dataclasses import dataclass

boundtype: t.TypeAlias = tuple[int, int]


def _independent(*bounds: boundtype) -> bool:
    """Checks if the input bounds are independent (has no overlap)"""
    for i, (a0, b0) in enumerate(bounds):
        for ap, bp in bounds[i+1:]:
            disjunct = bp < a0 or ap > b0
            if not disjunct:
                return False
            #
        #
    
    return True


@dataclass(init=False)
class IDSet:
    """Represents a set of freshness IDs, stored as disjunct tupes of upper and lower bounds
    of ranges.
    Supports boolean lookup like
    if 42 in idset_instance:
        ...
    """

    bounds: tuple[boundtype, ...]

    def __init__(self, *bounds: boundtype) -> None:
        # Make sure we're not trying to instantiate with bounds that overlap
        if not _independent(*bounds):
            raise ValueError(f"Bounds {bounds} are not disjunct")
        
        self.bounds = tuple(bounds)

    @classmethod
    def from_string(cls, s: str) -> IDSet:
        """Instantiate from a string like '<lower>-<upper>'."""
        a_str, b_str = s.strip().split("-")
        bound = (int(a_str), int(b_str))
        return cls(bound)
    
    def __contains__(self, id_: int) -> bool:
        """Check if an ID is contained in this instance"""
        return any(a <= id_ <= b for a, b in self.bounds)
    
    def __add__(self, other: IDSet) -> IDSet:
        """Combine with another instance.
        Returns another IDSet instance, with bounds that contain the set union of
        the intervals contained in the two instances"""

        # These are the bounds we keep for the resulting instance. These are always mutually independent
        use_bounds = list(self.bounds)

        for bound in other.bounds:
            a, b = bound
            for i in reversed(range(len(use_bounds))):
                # If disjunct, move on to compare against the next bound
                existing = use_bounds[i]
                if _independent((a, b), existing):
                    continue
                
                # Combine the bounds and remove the old one
                del use_bounds[i]
                ordered = sorted((a, b, *existing))
                a, b = ordered[0], ordered[-1]
            
            use_bounds.append((a, b))
        
        return IDSet(*use_bounds)

    def __len__(self) -> int:
        """Determines the number of IDs contained in this instance"""
        return sum(b - a + 1 for a, b in self.bounds)
    

def parse(s: str) -> tuple[list[IDSet], list[int]]:
    id_sets_string, ids_str = s.split("\n\n")
    
    ids = [int(line) for line in ids_str.splitlines()]
    id_sets = [IDSet.from_string(line) for line in id_sets_string.splitlines()]

    return id_sets, ids


def solve(data: str) -> tuple[int|str, ...]:
    id_sets, ids = parse(data)

    # Count up IDs that are present in any of the fresh sets
    star1 = sum(any(id_ in id_set for id_set in id_sets) for id_ in ids)
    print(f"Solution to part 1: {star1}")

    # Compute the union of all the fresh sets and determine its size
    r_combined = sum(id_sets, IDSet())
    star2 = len(r_combined)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
