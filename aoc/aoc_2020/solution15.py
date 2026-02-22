# *· `·.`*.· •  · ·      * `    ··.  `*  ·+. ·`• ·*      `·+  .   ·*  +·    •··`
# ·`·.+·    * .· *   ·  `.·* Rambunctious Recitation   •  +.··  *  `  ·.* ·•·`  
# · *  +· ·   •  ·  `  https://adventofcode.com/2020/day/15   ··`  .+. * · ·`. ·
# .·` . *`+ ·       ·.·`  ·.   · * .  `· •.   ·* .•· ·     ·  `. ·*·     ` ·.**·

import numba
import numpy as np
from numpy.typing import NDArray

_dtype = np.int64


def parse(s: str) -> NDArray[np.int_]:
    numbers = [int(elem) for elem in s.split(",")]
    res = np.array(numbers, dtype=_dtype)
    return res


@numba.njit(cache=True)
def run_game(numbers: NDArray[_dtype], stopat: int) -> int:
    """Runs the specified number of steps of the memory game, using
    the input starting numbers."""

    # Store times since last spoken in a preallocated array (much faster than dicts)
    last_spoken = np.full(stopat, -1, dtype=_dtype)
    n_initial_numbers = len(numbers)

    # Handle the initial numbers separately
    for i0 in range(n_initial_numbers):
        last_spoken[numbers[i0]] = i0 + 1
    
    number = numbers[-1]
    lastnumber = numbers[-2]

    for i in range(n_initial_numbers, stopat):
        lastnumber = number
        since_last = last_spoken[lastnumber]
        if since_last == -1:
            number = 0
        else:
            age = i - since_last
            number = age

        last_spoken[lastnumber] = i
    
    return number


def solve(data: str) -> tuple[int|str, ...]:
    numbers = parse(data)

    star1 = run_game(numbers, stopat=2020)
    print(f"Solution to part 1: {star1}")

    star2 = run_game(numbers, stopat=30_000_000)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 15
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
