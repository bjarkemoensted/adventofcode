#  ·•`. ·*    `·     .+*`·  ` · * `.     ·*`.    ·  ``* · +`  .*`·*·`  `   `·. *
# `..*· ` ·    *  .··*` .+    Springdroid Adventure ·*.    ·`· .*•`  ·`+.·*   .·
# *•.·`   *`·   * · .  https://adventofcode.com/2019/day/21  `·+ `·.  *· .`.•· `
# . ·   *` .  ·`·`+. ·*· `* . .  ·    ·`.*  `·`+. `    . *  · *`·. •·    `·*``·.

from aoc.aoc_2019.intcode import Computer


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def ascii_to_int(s: str) -> tuple[int, ...]:
    res = tuple(ord(char) for char in s)
    return res


def ints_to_ascii(*vals: int) -> str:
    return "".join(map(chr, vals))


# First springdroid code: D & ~(A & B & C)
walk_program = (
    "NOT A J",  # start by setting J to ABC
    "NOT J J",
    "AND B J",
    "AND C J",
    "NOT J J", # require not all of ABC, and D (landing spot)
    "AND D J",
    "WALK",
)

# Second code: D & (~A | ~B | ~C & (E | H))
run_program = (
    "NOT H J",  # if jumping over C, missing E and H would force a bad jump
    "NOT J J",
    "OR E J",
    "NOT C T",
    "AND T J",
    "NOT B T",  # also jump if no ground on A or B
    "OR T J",
    "NOT A T",
    "OR T J",
    "AND D J",  # always require a landing spot at D
    "RUN",
)


def run_instructions(program: list[int], instructions: tuple[str, ...]) -> int:
    """Run the springdroid instructions. Returns its damage report number if
    succesful. Otherwise, raise an error with the droid's ASCII output."""
    computer = Computer(program)
    for instruction in instructions:
        input_ = ascii_to_int(instruction+"\n")
        computer.add_input(*input_)

    out = []
    # Keep running as long as the computer generates output
    computer.run()
    while computer.stdout:
        out += list(computer.read_stdout(-1))
        computer.run()
    
    # If succesful, the final output is large, outside ascii range
    got_result = out[-1] > 127
    if got_result:
        res = out.pop()
        return res
    else:
        # Display error message if unsuccesful
        msg = ints_to_ascii(*out)
        raise RuntimeError(msg)
    

def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)
    
    star1 = run_instructions(program, walk_program)
    print(f"Solution to part 1: {star1}")

    star2 = run_instructions(program, run_program)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 21
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
