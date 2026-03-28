# `.·+    ··`+ `*  ·     .·`* ·       .`+· ·   * +·`  ·`  .+· .     •· `  *.· ·.
# .*`··+  .    ·  + `.·*      `·  Dumbo Octopus    . · · * `+ ·   .   +·` `*.·`·
# .`·`  . ` · ·*  ·    https://adventofcode.com/2021/day/11  .`· *    ·  `.·` *·
# ··`.  `·    .  ·`+*· .`·+*    ·   `.*·     `*    ··`*.    `· *  ·   ` ·.· * ·`

from functools import cache

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> NDArray[np.int_]:
    res = np.array([[int(s) for s in line.strip()] for line in s.splitlines()], dtype=int)
    return res


@cache
def get_neighborhood_coords(shape: tuple[int, int], i: int, j: int) -> list[tuple[int, int]]:
    """Takes a matrix (2d np array) shape and returns a list of coordinates of neighbors.
    Returns 8 points (E, W, S, N, SW etc), unless input is at the edge/corner."""
    incs = (-1, 0, +1)
    # Generate all 8 surrounding points
    cands = ((i + di, j + dj) for di in incs for dj in incs if not (di == dj == 0))

    # Keep those that aren't out of bounds
    res = [(ip, jp) for ip, jp in cands if all(0 <= c < lim for lim, c in zip(shape, (ip, jp), strict=True))]
    return res


def step(m: NDArray[np.int_]) -> int:
    # Increment all energy levels by one
    inc = np.ones(shape=m.shape, dtype=int)
    m += inc

    # We're done if all energy levels are below the flash threshold
    # Find those about to flash
    already_flashed = set([])
    about_to_flash = set([(i, j) for i, j in np.ndindex(m.shape) if m[i, j] >= 10])

    while about_to_flash:
        # Matrix to keep track of which elements to increment
        blast = np.zeros(shape=m.shape, dtype=int)
        for i, j in about_to_flash:
            # Update set of octopi that already flashed
            already_flashed.add((i, j))
            # Increment the light blast in the vicinity of each such octopus
            for ii, jj in get_neighborhood_coords(m.shape, i, j):
                blast[ii, jj] += 1
            #

        # Flash neighbors
        m += blast

        # Find those about to flash again. Done if none.
        remaining_coords = set(np.ndindex(m.shape)) - already_flashed
        about_to_flash = set([(i, j) for i, j in remaining_coords if m[i, j] >= 10])

    # Remove energy from those who flashed
    n_flashed = len(already_flashed)
    for i, j in already_flashed:
        m[i, j] = 0

    return n_flashed


def count_flashes(M: NDArray[np.int_], n_rounds=100) -> int:
    M = M.copy()
    # Count how many luminescent octopi flash during 100 iterations
    res = 0

    for _ in range(100):
        n_flashes_this_round = step(M)
        res += n_flashes_this_round
    return res


def determine_first_synchronized_round(M: NDArray[np.int_]) -> int:
    M = M.copy()
    n = 0
    while not all(v == 0 for v in M.flat):
        step(M)
        n += 1

    return n


def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)

    star1 = count_flashes(M)
    print(f"Solution to part 1: {star1}")    

    star2 = determine_first_synchronized_round(M)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 11
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
