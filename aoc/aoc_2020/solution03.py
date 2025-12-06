# · ` · *+ `.·*  ` .  • · `·   `    ··`. * +   *·    · ` *.`* ·  ·   *.   * ·.·`
#  `  ` ·.  ·.`  *·   `*  ·•   Toboggan Trajectory `*  ·` ·  *•`  · ·` . · +`·.*
# ··.* ·` ·  • · .  ·  https://adventofcode.com/2020/day/3  ·  ·`.    *` `.· *·.
# `.·` *.· *`   · `  +     *`·.   ·  *. `· ·  .`*`·  *· ·.   ·  .*`  · ·. ·`  •·

from functools import reduce

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line.strip()) for line in s.splitlines()])
    return res


def scan_trajectory(map_: NDArray[np.str_], direction: tuple[int, int]) -> int:
    """Follows a path given by integer increments of the input direction vector, along the
    input ASCII map. Counts the number of trees encountered"""
    
    charmap = {"#": 1, '.': 0}
    i, j = 0, 0

    di, dj = direction
    assert dj > 0

    height, width = map_.shape
    n_trees = charmap[map_[i, j]]

    while i < height-1:
        i = i + dj
        j = (j + di) % width

        char = map_[i, j]
        n_trees += charmap[char]

    return n_trees


def solve(data: str) -> tuple[int|str, ...]:
    map_ = parse(data)

    path = (3, 1)
    star1 = scan_trajectory(map_, path)
    print(f"Solution to part 1: {star1}")

    paths = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
    star2 = reduce(lambda a, b: a*b, (scan_trajectory(map_, p) for p in paths))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
