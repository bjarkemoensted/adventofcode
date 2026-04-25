# .·*`·   ·. · *·`   .+·  ` • ·   · +`    ·`·   *  `.    · ` +·  *.`·   ··*  ·`.
# ·.`*  *·`·.  ·  ·.*        ·  ·  Sensor Boost ·. · ` •· ` .  *` · `·· +`·. ·• 
#  `·+·· `. ·  .*·  `· https://adventofcode.com/2019/day/9 •·  `.·`·+     .· `··
# ·  ·` ·*  `.·   .· *  ` ··    ` * · ·• `..*·  `   ·· *·`   .*·    ·+`*. ` ·  ·

from aoc.aoc_2019.intcode import Computer


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)

    star1 = Computer(program).add_input(1).run().read_stdout()
    print(f"Solution to part 1: {star1}")

    star2 = Computer(program).add_input(2).run().read_stdout()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 9
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
