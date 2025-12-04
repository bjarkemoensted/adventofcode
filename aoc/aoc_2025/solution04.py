# `··. •`· ·*  .·  .`*· ` . ·•   ·  `    ·+  · ` .*· `.` ·       ·  · `.·*· ·``.
# .`*· `·.·• *`      ·   .·+   Printing Department  ·   .  ·*.  `  ·  ·   ..* ··
# .· +·  `.  ·    `  . https://adventofcode.com/2025/day/4   ·. ·      ·•`   ·. 
# ·* `+ .·``     ·.· ` +·*   `.   ·· `    ·*·.    · `    .·.   ·  ·`   • · ·`.·*


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
import typing as t

coordtype: t.TypeAlias = tuple[int, int]

_occupied = "@"


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line) for line in s.splitlines()])
    return res


def count_occupied_neighbors(M: NDArray[np.bool_]) -> NDArray[np.int_]:
    """Returns an array where each element is the number of neighboring sites
    occupied by paper"""
    p = np.pad(M, 1, constant_values=(False,))
    windows = sliding_window_view(p, (3, 3))
    res = windows.sum(axis=(-2, -1)) - M
    
    return res


def repeated_cleanup(M_occupied: NDArray[np.bool_], n_iterations=-1) -> int:
    """Repeatedly removes occupied sites with n < 4 adjacent occupied sites.
    Returns the number of sites cleaned in this way.
    Optionally, a number of iterations can be provided. Otherwise, the procedure
    continues until no more sites can be removed."""
    
    M_occupied = M_occupied.copy()
    res = 0
    nits = 0

    while True:
        # Stop i n iterations is exceeded
        nits += 1
        if n_iterations != -1 and nits > n_iterations:
            break
        
        # Figure out which occupied sites to remove
        neighborhood_counts = count_occupied_neighbors(M_occupied)
        remove = np.argwhere(M_occupied & (neighborhood_counts < 4))
        if not remove.any():
            break
        
        # Increment counter and update array with remaining occupied sites
        res += len(remove)
        for i, j in remove:
            M_occupied[i, j] = False
        #

    return res


def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)
    M_occupied: NDArray[np.bool_] = M == _occupied

    star1 = repeated_cleanup(M_occupied, n_iterations=1)
    print(f"Solution to part 1: {star1}")

    star2 = repeated_cleanup(M_occupied)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 4
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
