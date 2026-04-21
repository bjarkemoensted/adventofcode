# ·`. · *·. ` . * · ·   . *· .   · `* .  +   · ·.`· *+ `··    *.··· *` .· *.·* ·
# *  · `  · .`  ·   .`*·+. `· * 1202 Program Alarm +. · *   · ·  `+.·  ·*.     `
# `·+.*+·`       ·     https://adventofcode.com/2019/day/2 *+  ·  `  . * ·`·* ··
# ·  * ·.· +.· .`   ··+ ·   .   ·`*·.·+ ` · • · +.*  ·•   .·`·   *   ·  .`· `.·*

from aoc.aoc_2019.intcode import Computer


def parse(s: str) -> list[int]:
    res = list(map(int, s.split(",")))
    return res


def compute_output(program: list[int], noun: int, verb: int) -> int:
    """Computes the output when running the program with the specified noun+verb"""
    program = program.copy()
    program[1] = noun
    program[2] = verb
    computer = Computer()
    res = computer.run(program)
    return res


def brute_force_output(program: list[int], target: int) -> tuple[int, int]:
    """Brute forces the noun and verb required to get the specified output from the program"""
    for noun in range(100):
        for verb in range(100):
            output = compute_output(program.copy(), noun, verb)
            if output == target:
                return noun, verb
            #
        #
    
    raise RuntimeError


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)

    star1 = compute_output(program, 12, 2)
    print(f"Solution to part 1: {star1}")

    noun, verb = brute_force_output(program, target=19_690_720)
    star2 = 100*noun + verb
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 2
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
