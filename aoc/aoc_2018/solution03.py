# `·*·  `   ·.` ·*.   · `   .* ··   · .    ·*.`· ·   `  .·  * · · .. *`  ·  .·+·
# ·` + *..·  `   ··*·.   `  No Matter How You Slice It  · ·.`  .`  *.·     ·• `.
#  *·.`· `.· * ·`    · https://adventofcode.com/2018/day/3  .• ·   ·  · .*`   · 
# `·`  . ·+  · .·`*· * .·•·  ··`     ` +·.   * ·.`  · .    `·*.  · `   +·  .·`· 

import numpy as np
from typing import cast, Iterable, TypeAlias


claim: TypeAlias = tuple[tuple[int, int], tuple[int, int]]


def parse(s: str) -> dict[int, claim]:
    res = dict()
    for line in s.splitlines():
        a, _, b, c = line[1:].split()
        id_ = int(a)
        offsets = tuple(int(elem) for elem in b[:-1].split(","))
        lens = tuple(int(elem) for elem in c.split("x"))
        res[id_] = cast(claim, (offsets, lens))
    
    return res


def count_overlaps(claims: Iterable[claim], size=1000) -> np.typing.NDArray[np.int_]:
    """Build a matrix of overlap counts, so cell values represent the number of overlapping claims"""
    
    # Start with a coutn of zero overlaps everywhere
    m = np.zeros((size, size), dtype=int)
    
    # Add 1s in a rectangle as specified by each claim
    for (j, i), (dj, di) in claims:
        m[i:i+di, j:j+dj] += 1
    
    return m


def solve(data: str) -> tuple[int|str, int|str]:
    claims = parse(data)

    m = count_overlaps(claims.values())
    # Count the number of cells where one or more claims overlap
    star1 = sum(v > 1 for v in m.flat)
    print(f"Solution to part 1: {star1}")
    
    # The completely distinct claim must be where the one where all counts are equal to 1
    star2 = next(id_ for id_, ((j, i), (dj, di)) in claims.items() if np.all(m[i:i+di, j:j+dj] == 1))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()