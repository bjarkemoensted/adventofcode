# ·`.·   .`  ·  · `. .·`·  .*·   ` .  ·.  ··  `. `+·  * ·` ·  ·   · *·` . `+  ·.
# `·+ `.·    .·*`.  · `. Sunny with a Chance of Asteroids       . ` · ·   .·`•.`
# .··`+·`  *· •·.` ·.  https://adventofcode.com/2019/day/5  .` ·   .  +·`·  ·.*`
#  . ·.`•`.·  `+ · `  .· • `.+· · `  •· ..·  `· .  *   ·  `.·   ·  · * `·`·. `··

from aoc.aoc_2019 import intcode


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)
    
    star1 = intcode.Computer(program).add_input(1).run().stdout[-1]
    print(f"Solution to part 1: {star1}")

    star2 = intcode.Computer(program).add_input(5).run().stdout[-1]
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
