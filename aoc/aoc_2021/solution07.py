# *·` . ·`   *.   · `.· •·*   `·+    .··  *`·  ·.  `*· `. ·    *`  ··. * ·`.*`·.
# · .  ·    ·  * .`.·•  · ·` The Treachery of Whales   · . `.   *·` . ··..*·`• ·
# ·`*·+  *.· ` .·    ` https://adventofcode.com/2021/day/7   `   `·.    *`.·.  `
# .··   . ·   *·     ·•`    `·•· *`.··      `    ·.*. ·`    · · . +     ` ·`·.* 

from typing import Callable


def parse(s: str) -> list[int]:
    res = [int(s) for s in s.strip().split(",")]
    return res


def _default_metric(a: int, b: int) -> int:
    return abs(a - b)


def dist(arr: list[int], target: int, metric: Callable[[int, int], int]|None=None) -> int:
    """Computes the sum of distances to target, given a distance metric"""
    if metric is None:
        metric = _default_metric
    res = sum(metric(val, target) for val in arr)
    return res


def dist_quad(a: int, b: int) -> int:
    """Metric for when traveling a distance of x costs 1 + 2 + ... + n."""
    d = abs(a - b)
    res = int(d * (d + 1) / 2) # Gauss trick: 1+2+...+n = n(n+1)/2
    return res


def solve(data: str) -> tuple[int|str, ...]:
    numbers = parse(data)
    # List potential target values where the crab submarines can line up
    targets = range(min(numbers), max(numbers))

    # Find the shortest distance
    best_target = min(targets, key=lambda x: dist(numbers, x))
    star1 = dist(numbers, best_target)
    print(f"Solution to part 1: {star1}")

    # Compute shortest distance using the quadratic metric thingy.
    best_target2 = min(targets, key=lambda x: dist(numbers, x, dist_quad))
    star2 = dist(numbers, best_target2, dist_quad)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 7
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
