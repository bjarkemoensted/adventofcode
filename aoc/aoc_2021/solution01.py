# *.` +. ·*· `·.    · • `  . *· ·`   .    · .·*   ·*.`   ·  ·. * ·  · `*  `.··`*
#  ·.    `·   *`.· . · *    .· · . Sonar Sweep ` .  · +·   ..`·   + ` .· .·`+   
# .`·.` *   `. ·    *. https://adventofcode.com/2021/day/1 *`·.    ·.    * · `·.
# · . ·`·*  ·`. ·     ` +* ·   ·.`· * `. · . *· ·  `    · .`·   * ·`+ ·.·`  ` .·


def parse(s: str) -> list[int]:
    res = [int(line) for line in s.splitlines()]
    return res


def solve(data: str) -> tuple[int|str, ...]:
    sweep = parse(data)

    star1 = sum(sweep[i+1] > sweep[i] for i in range(len(sweep) - 1))
    print(f"Solution to part 1: {star1}")

    star2 = sum(sweep[i+3] > sweep[i] for i in range(len(sweep) - 3))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 1
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
