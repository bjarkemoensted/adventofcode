# ·.  ·`+`·. * · . `*·` . *   *· .· ` +· ·   * ··`.       .·+  * · .   `.·  .*· 
# `* ·.. · * .   ·`    * +·`·    · Conway Cubes * `· .  · `*· +·    ·  ·`. *·.• 
# *·  `· .   · `  · .` https://adventofcode.com/2020/day/17   ·` .   · * `. +· ·
# · `      ·* `  + .·* ·` .  · ` .  *·  `  •· ·     ·  . `  .·* · +   · * ·. `*.

from collections import Counter
from itertools import product

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> list[tuple[int, int]]:
    """Parse the (2D) coordinates of the initially active cubes"""
    res: list[tuple[int, int]] = []
    active_symbol = "#"
    for i, line in enumerate(s.splitlines()):
        for j, char in enumerate(line):
            if char == active_symbol:
                res.append((i, j))
            #
        #

    return res


def expand(coords: tuple[int, ...], dim: int) -> tuple[int, ...]:
    """Expands coordinates to the specified number of dimensions by appending zeroes"""
    n_missing = max(0, dim - len(coords))
    res = coords + n_missing*(0,)
    return res


def compute_offsets(dim: int) -> NDArray[np.int_]:
    """Computes a 2D array of all offsets from a given site"""

    _offsets_single = (-1, 0, 1)
    shifts = dim*[_offsets_single]

    its = product(*shifts)
    rows = []
    for coords in its:
        # Skip all-zero offsets 
        if all(x == 0 for x in coords):
            continue
        rows.append(coords)

    m = np.array(rows, dtype=np.int64)

    return m


def cycle(coords: list[tuple[int, ...]], n_cycles: int) -> int:
    """Simulates n cycles of the conway cubes interacting.
    Returns the number of active cubes after the simulation"""

    dim = len(coords[0])
    assert all(len(c) == dim for c in coords)
    offsets = compute_offsets(dim)
    # Runnning set of the currently active cubes
    active_cubes = set(coords)

    for _ in range(n_cycles):
        # Count the number of active neighbors at each coordinate
        arr = np.array(list(active_cubes))
        # Compute all combinations of cube coordinate + offsets
        neighbors = (arr[:, None, :] + offsets[None, :, :]).reshape(-1, arr.shape[1])

        # Update the active set based no the rules
        counts = Counter(map(tuple, neighbors))
        active_cubes = {cube for cube, n in counts.items() if n == 3 or (cube in active_cubes and n == 2)}

    return len(active_cubes)


def solve(data: str) -> tuple[int|str, ...]:
    coords_2d = parse(data)
    n_cycles = 6

    coords_3d = [expand(c, 3) for c in coords_2d]
    star1 = cycle(coords_3d, n_cycles=n_cycles)
    print(f"Solution to part 1: {star1}")

    coords_4d = [expand(c, 4) for c in coords_2d]
    star2 = cycle(coords_4d, n_cycles=n_cycles)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 17
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
