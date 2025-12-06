# *.   `· ·. ·     * ·.``+.·   ·*. ·       ·  *`  · .   + .· ` ··•*`· . * ·` .  
#  ·+` ·  .·` +•. ·. *·    *·`  · Report Repair ·   *   ·    .·`*.`·  · .* · `*.
# ` . `  ·  .*  · `·+. https://adventofcode.com/2020/day/1  ·*`   ·.` * · `*  ··
# · ··.•    ·.`·*     `· *·  .·  `  · +· ·  `*•.`  ·+ · `· +.  • ·     ·`·*. ·•`

import itertools
from functools import reduce


def parse(s: str) -> list[int]:
    return [int(line) for line in s.splitlines()]


def find_elems_with_sum(numbers: list[int], target=2020, n_elems=2) -> int:
    """Given the input numbers, finds the first combination of n elements which sum to
    the target value, and returns their product.
    Loops over all combinations of n_elems-1 elements, and attempts to lookup the remaining
    value in the set of input numbers."""

    numbers_set = set(numbers)
    combinations = itertools.combinations(numbers, n_elems-1)

    for comb in combinations:
        missing_number = target - sum(comb)
        if missing_number in numbers_set:
            return reduce(lambda a,b: a*b, comb, missing_number)
        #
    
    raise RuntimeError("Sum not found")


def solve(data: str) -> tuple[int|str, ...]:
    numbers = parse(data)

    star1 = find_elems_with_sum(numbers)
    print(f"Solution to part 1: {star1}")

    star2 = find_elems_with_sum(numbers, n_elems=3)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 1
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
