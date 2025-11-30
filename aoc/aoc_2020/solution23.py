# ·.·`     `·+  .·· `+·*`·     · +`  ·.   ·+ ·.·`  ·  .`*·   ··+.`· ·*`  .·*· `·
# .· ··`.      *  `· · ·.  +·· .* · Crab Cups ·. `·    ··+.`·*.  ·* `·.· ·  ·*.·
# · •.`*`· ·* ·.    ·` https://adventofcode.com/2020/day/23 ·  .   ··`*+·    .·.
#  ·· *·    `·  ` .  ·*` +··    .`   ·*·    · *  . ·`. *·  ·•`  ·. ·  ·* +.·`· *


def parse(s: str) -> object:
    res = s  # TODO parse input here
    return res


def solve(data: str) -> tuple[int|str, ...]:
    parsed = parse(data)

    # TODO solve puzzle
    star1 = None
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
