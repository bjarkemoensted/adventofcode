# · ` .·  `  ·`*.  .*· `·*+·.`. `   .·  ·.·  ·*   `·.    ·`·•.`  ·*·   `+   `·.•
# .·. ·`  ·    ` ·· . ·• . `    .* Space Police  · •`·    ·* `·.·  +· `.· ·. ·  
#  ` . *·.  ·` ··*`  . https://adventofcode.com/2019/day/11  · *  · `.··` .`·+*·
#  .·`   ·.*` ·   · `  .    ··`  · *`·+.        ·`* *  · ` .·  ·      * .·`·  ·`

from enum import IntEnum

import numpy as np
from aococr import aococr

from aoc.aoc_2019.intcode import Computer


class Colors(IntEnum):
    BLACK = 0
    WHITE = 1


class Turns(IntEnum):
    LEFT = 0
    RIGHT = 1


# up, right, down, left
directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def paint(program: list[int], starting_color=Colors.BLACK) -> dict[tuple[int, int], int]:
    computer = Computer(program)
    res: dict[tuple[int, int], int] = dict()
    
    pos = np.array([0, 0], dtype=int)
    i0, j0 = pos
    res[(i0, j0)] = starting_color
    dir_ind = 0
    dir_shifts = {Turns.RIGHT: +1, Turns.LEFT: -1}

    while not computer.halted:
        i, j = pos
        input_ = res.get((i, j), Colors.BLACK)
        computer.add_input(input_).run()
        new_color = computer.read_stdout()
        res[(i, j)] = new_color
        turn_direction = Turns(computer.read_stdout())

        dir_ind = (dir_ind + dir_shifts[turn_direction]) % len(directions)
        pos += directions[dir_ind]

    return res


def determine_identifier(pixels: dict[tuple[int, int], int]) -> str:
    """Given the input pixel color values, determines the registration code for the
    pattern formed by the pixels"""
    
    # Make an array with the coordinates of the white panels
    pixel_coords = np.vstack([np.array(k) for k, v in pixels.items() if v == Colors.WHITE])
    # Shift to ensure coords are non-negative
    pixel_coords -= pixel_coords.min(axis=0)
    
    # Make an array with the pattern painted by the robot
    height, width = pixel_coords.max(axis=0) + 1
    hull = np.full(shape=(height, width), fill_value=Colors.BLACK, dtype=int)
    hull[*pixel_coords.T] = Colors.WHITE

    # Read the characters formed by white panels
    res = aococr(hull, pixel_on_off_values=(Colors.WHITE, Colors.BLACK))
    return res


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)

    painted = paint(program)
    star1 = len(painted)
    print(f"Solution to part 1: {star1}")

    painted_correct = paint(program, starting_color=Colors.WHITE)
    star2 = determine_identifier(painted_correct)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 11
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
