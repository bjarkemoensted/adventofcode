# ꞏ    . ⸳•    •` *   ` ⸳*`. + • ⸳•.   ⸳ `* . ꞏ` . •*⸳⸳   .`     ` *⸳   ꞏ*. ⸳  .
# .*  ` • ⸳. +     + ⸳ꞏ  .*   * Historian Hysteria   . **⸳  `   .•       . ⸳•`  
# .  `*           ⸳ .  https://adventofcode.com/2024/day/1   ⸳      * ⸳`ꞏ  + ⸳ +
# +⸳.     `.*  .⸳   +`  . ⸳ *`  .   * `*  .   •.ꞏ•⸳ `   . *     ⸳ * . ꞏ.  `  *⸳ 

from collections import Counter


def parse(s):
    pairs = [tuple(int(part) for part in line.split()) for line in s.splitlines()]
    a, b = zip(*pairs)
    return a, b


def dist(left, right):
    """Pairs up the sorted elements from the lists and sums the absolute diffs (1-D Manhatten dist)"""
    res = sum(abs(x - y) for x, y in zip(*(sorted(list_) for list_ in (left, right))))
    return res


def similarity(left, right):
    """Sums the products of each number in left list times number of times it occurs in right list"""
    counts = Counter(right)
    res = sum(counts.get(num, 0)*num for num in left)
    return res


def solve(data: str):
    left, right = parse(data)

    star1 = dist(left, right)
    print(f"Solution to part 1: {star1}")

    star2 = similarity(left, right)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2024, 1
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
