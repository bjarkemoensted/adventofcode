# · ``  ··* .`*  .  · ·  *`· `.·•   `+.·*  .·`· .      +·`•.··  .+`·      ·.`+*·
# ·` · . *`· ·  . +` · *`· .·  *·` Sea Cucumber ·*     · *  . ·  `·  `+.·`   · .
# . ·. ·`   `*·  +.* ` https://adventofcode.com/2021/day/25 +    .* `  ·.·`+  `·
#  ·. ·**`·.   ` `· *. `* ·    .*·`·. ·  `·* .*   ·`·+      `•·* · `..··`* .·•· 

import numpy as np
from numpy.typing import NDArray

_empty = "."
# sea cucumber symbols and their direction vectors
_order = (
    (">", np.array([0, 1])),
    ("v", np.array([1, 0])),
)


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line) for line in s.splitlines()])
    return res


class Seabed:
    """Represents the seabed with current sea cucumber positions.
    The type of cucmber is represented by the index of the order in which
    the sea cucumber's herd will attempt moving. Empty sites are represented
    by -1."""

    def __init__(self, map_: NDArray[np.str_]) -> None:
        # Convert the map into a numerical representation
        inv = {char: i for i, (char, _) in enumerate(_order)}
        self.m = np.empty(shape=map_.shape, dtype=int)
        for (i, j), val in np.ndenumerate(map_):
            self.m[i, j] = -1 if val == _empty else inv[val]
        
        self.shape_arr: NDArray[np.int_] = np.array(self.m.shape)

    def display(self) -> None:
        """Helper method for printing the current state"""
        lines = ["".join(_empty if val == -1 else _order[val][0] for val in row) for row in self.m]
        print("\n".join(lines))
    
    def tick(self) -> bool:
        """Updates the sea cucumber locations.
        Returns a boolean indicating whether any positions were updated."""

        updated = False
        for ind, (_, delta) in enumerate(_order):
            # Find the current locations of members of the herd we're updating
            inds = np.argwhere(self.m == ind)
            # Determine where they want to move, and whether the site is free
            shifted = inds + delta
            targets = shifted % self.shape_arr
            isfree = self.m[*targets.T] == -1
            moveinds = inds[isfree].T
            if not updated:
                updated = any(isfree)
            
            # For those that can move, remove from old site and add to new site
            self.m[*moveinds] = -1
            self.m[*targets[isfree].T] = ind
        
        return updated

    def update_repeated(self) -> int:
        """Repeatedly update positions, until no sea cucumbers move.
        Returns the number of iterations used (including the final update-attempt,
        in which nothing moved)."""
        n = 1
        while self.tick():
            n += 1
        
        return n


def solve(data: str) -> tuple[int|str, ...]:
    map_ = parse(data)
    seabed = Seabed(map_)
    
    star1 = seabed.update_repeated()
    print(f"Solution to part 1: {star1}")

    return star1, -1


def main() -> None:
    year, day = 2021, 25
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
