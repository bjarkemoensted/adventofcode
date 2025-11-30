# `·.· +.·``+· ·.  ·  * ` .*    · ·. •· `  ·.   *`  ·.    ·*  `*  .·  .·*    `+·
# *.``+·  . ·*   ·. ·+ ` ·*·` .   Adapter Array  · `. *  · `· +.• ·` *·. .`· ·`*
# ·*·. ` ` ·`  ` .· .  https://adventofcode.com/2020/day/10    ·.` . ·• `· `* ·.
#  ` ·`*·. `.   ·•   ·  +`·.  `·*. *`· .·    *  . · + ·`  *  .· `   ·`  .*·+· . 

from collections import Counter
from itertools import chain, combinations
from typing import Iterator


def parse(s: str) -> list[int]:
    res = [int(line) for line in s.splitlines()]
    return res


def count_differences(ratings: list[int]) -> dict[int, int]:
    """Count the differences between each subsequent pair of ratings"""
    diffs = (ratings[i+1] - ratings[i] for i in range(len(ratings) - 1))
    counts = Counter(diffs)

    return counts


def chop(arr: list[int], dist_cut=3) -> Iterator[list[int]]:
    """Breaks input numbers into parts, breaking every time the specified distance
    between two subsequent elements is encountered"""
    buffer = [arr[0]]
    for elem in arr[1:]:
        if elem - buffer[-1] == dist_cut:
            yield buffer
            buffer = []
        buffer.append(elem)
        #
    if buffer:
        yield buffer


def make_subsets(arr: list[int]) -> Iterator[tuple[int, ...]]:
    """Returns a generator of all subsets of the input values."""
    combs = (combinations(arr, i) for i in range(len(arr) + 1))
    subs = chain.from_iterable(combs)
    return subs


def brute_force(chunk: list[int]) -> int:
    """Counts the number of valid ways we could rearrange the elements in the input,
    except at the beginning and end."""
    if len(chunk) <= 2:
        return 1

    middle = chunk[1:-1]
    subsets = make_subsets(middle)
    res = 0
    for subset in subsets:
        temp = [chunk[0]] + list(subset) + [chunk[-1]]
        valid = all(temp[i+1] - temp[i] <= 3 for i in range(len(temp) - 1))
        res += valid
    return res


def count_combinations(ratings: list[int]) -> int:
    """Counts the number of ways to rearrange"""
    res = 1
    cache: dict[int, int] = dict()
    chunks = chop(ratings)

    for chunk in chunks:
        n_components = len(chunk)
        try:
            res *= cache[n_components]
        except KeyError:
            perm = brute_force(chunk)
            cache[n_components] = perm
            res *= perm
        #
    return res


def solve(data: str) -> tuple[int|str, ...]:
    ratings = parse(data)

    ordered = sorted(ratings)
    ordered = [0]+ordered+[ordered[-1]+3]

    diff_counts = count_differences(ordered)
    star1 = diff_counts[1]*diff_counts[3]
    print(f"Solution to part 1: {star1}")

    star2 = count_combinations(ordered)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 10
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
