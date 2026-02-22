# `· `.   *  ··.`   ·  ·` ·*  .    ·   ·   *   ·.·* ·`  ` · .* .   `.•·· *` .+·`
# ·.•.·*·     *· `  +· `  •.· *·*.· Rain Risk `.  · •· ` · .*     ` ·*. · ·`. `·
# ·.`·  `*·.·  `·•·    https://adventofcode.com/2020/day/12 ·`·  * ·     . · *·.
# .+· `·. `· .     `·. • `  .*· ·   `.· .+`    *·+ .  +· `   ·.  ·   ·+• ` .··.`

import typing as t

import numpy as np
from numpy.typing import NDArray

instype: t.TypeAlias = t.Literal["N", "E", "S", "W", "L", "R", "F"]


compass = {
    "N": np.array([0, 1]),
    "S": np.array([0, -1]),
    "E": np.array([1, 0]),
    "W": np.array([-1, 0])}

turns = {
    "L": +1,
    "R": -1}


def _is_instype(s: str) -> t.TypeGuard[instype]:
    return s in t.get_args(instype)


def parse(s: str) -> list[tuple[instype, int]]:
    res: list[tuple[instype, int]] = []
    for line in s.splitlines():
        instruction = line[0]
        assert _is_instype(instruction), str(instruction)
        val = int(line[1:])
    
        res.append((instruction, val))

    return res


def rotate(vec: NDArray[np.int_], angle: int) -> NDArray[np.int_]:
    """Rotates the input vector by the specified angle"""

    theta = np.deg2rad(angle)
    M = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    new = M.dot(vec)
    res = np.rint(new).astype(int)

    return res


class Ferry:
    def __init__(self, position: tuple[int, int]=(0, 0), direction: tuple[int, int]=(1, 0)) -> None:
        self.position = np.array(position)
        self.direction = np.array(direction)

    def run_instruction(self, inst):
        action, val = inst
        if action in compass:
            self.position += compass[action]*val
        elif action == "F":
            self.position += self.direction*val
        elif action in turns:
            angle = turns[action]*val
            new_direction = rotate(self.direction, angle=angle)
            self.direction = new_direction
        #
    #


class WeirdFerry:
    def __init__(self, waypoint: tuple[int, int], position: tuple[int, int]=(0, 0)):
        self.position = np.array(position)
        self.waypoint = np.array(waypoint)

    def run_instruction(self, inst):
        action, val = inst
        if action in compass:
            self.waypoint += compass[action]*val
        elif action == "F":
            self.position += self.waypoint*val
        elif action in turns:
            angle = turns[action]*val
            new_waypoint = rotate(self.waypoint, angle=angle)
            self.waypoint = new_waypoint
        #
    #


def solve(data: str) -> tuple[int|str, ...]:
    instructions = parse(data)

    ferry = Ferry()

    for instruction in instructions:
        ferry.run_instruction(instruction)

    star1 = sum(abs(val) for val in ferry.position.flat)
    print(f"Solution to part 1: {star1}")

    waypoint = (10, 1)
    wf = WeirdFerry(waypoint)

    for instruction in instructions:
        wf.run_instruction(instruction)

    star2 = sum(abs(val) for val in wf.position.flat)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 12
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
