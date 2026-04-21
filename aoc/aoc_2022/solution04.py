# `·  ·. ·.`·     ·  ·  * · .  · .` ·*  ·   ·`     * ·.      ·*. ·   . · `  · * 
# ·*·`. · `.•  ·   `. ·`     * .·  Camp Cleanup ·. .      ·`.  * ·   ·.`·   *`··
# *.  ·  ·  ` *.`·   * https://adventofcode.com/2022/day/4  ·. · `    ·   · .·` 
# ·`.· ·.    `·*    · * `. ·   ` + .*`· .   ·*· . ·  `· •.`    ` •·   `· * · +·.


def parse(s: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    tups = []
    for line in s.splitlines():
        ((a, b), (c, d)) = [tuple(map(int, elem.split("-"))) for elem in line.strip().split(",")]
        tups.append(((a, b), (c, d)))

    return tups


def tuple_contains_other(tup1: tuple[int, int], tup2: tuple[int, int]) -> bool:
    a1, b1 = tup1
    a2, b2 = tup2

    return a1 <= a2 and b1 >= b2


def tuple_have_overlap(tup1: tuple[int, int], tup2: tuple[int, int]) -> bool:
    a1, b1 = tup1
    a2, b2 = tup2

    return b1 >= a2 and not a1 > b2


def solve(data: str) -> tuple[int|str, ...]:
    parsed = parse(data)

    star1 = 0
    star2 = 0
    for tup1, tup2 in parsed:
        star1 += tuple_contains_other(tup1, tup2) or tuple_contains_other(tup2, tup1)
        star2 += tuple_have_overlap(tup1, tup2) or tuple_have_overlap(tup2, tup1)

    print(f"Solution to part 1: {star1}")
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 4
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
