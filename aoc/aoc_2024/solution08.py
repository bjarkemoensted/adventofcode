# `·   • · ·  .· `   .·   `·  •.·   ·  + `· * ` ·+    ·   *· . `* ·.    ·`  .· ·
# *·` .·· *  .  *·  ` *  ·  · Resonant Collinearity  · `* `  · ·  .   *  ··`  .+
# ·*. ·. `·*    ·.     https://adventofcode.com/2024/day/8   ` *··    . * `   ·.
# .`··` .  .+·`   * .·  ` ·  · ·       ·`* ·     .  ·* ·`      .` •· `·+  . · *`


from collections import defaultdict

import numpy as np


def parse(s: str):
    res = np.array([list(line) for line in s.splitlines()])
    return res


class Grid:
    """Grid class for keeping track of antenna locations and calculating antinodes"""

    def __init__(self, m: np.ndarray):
        # Keep a reference to the input array
        self.m = m
        
        # Map each frequency to a set of coordinates with antennas using that frequency
        self.d = defaultdict(lambda: set([]))
        for i, j in np.ndindex(self.m.shape):
            char = self.m[i, j]
            if char != ".":
                self.d[char].add((i, j))
            #
        #
    #

    def _in_bounds(self, crd):
        return all (0 <= x < dim for x, dim in zip(crd, self.m.shape))

    def get_antinodes(self, frequency: str, with_resonance=False):
        """Returns a set of the coordinates of antinodes at the given frequency.
        If with_resonance, takes resonant harmonics^tm into account."""

        res: set[tuple[int, ...]] = set([])
        coords = sorted(self.d[frequency])
        
        for x1 in coords:
            if with_resonance:
                # Also add the antenna's coordinates
                res.add(x1)
            for x2 in coords:
                if x1 == x2:
                    continue
                # Find the displacement between each pair of antennas
                delta = tuple(b - a for a, b in zip(x1, x2))
                
                # Step in increments of the displacement vector until falling off the map
                an: tuple[int, ...] = x2
                while self._in_bounds(an := tuple(a + b for a, b in zip(delta, an))):
                    res.add(an)
                    if not with_resonance:
                        # If running without resonance, stop after a single step
                        break
                    #
                #
            #
        #
        
        return res
    
    def get_all_antinodes(self, with_resonance=False):
        """Returns a set of the antinodes for all frequencies"""

        nodes = set([])
        freqs = self.d.keys()
        for freq in freqs:
            nodes |= self.get_antinodes(frequency=freq, with_resonance=with_resonance)
        
        return nodes
    #


def solve(data: str) -> tuple[int|str, int|str]:
    parsed = parse(data)
    grid = Grid(m=parsed)

    star1 = len(grid.get_all_antinodes())
    print(f"Solution to part 1: {star1}")

    star2 = len(grid.get_all_antinodes(with_resonance=True))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 8
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
