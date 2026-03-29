#  · ·`     ·  ·*`  * ·  *·   ·  * ·• .·` ·` .*     `·   ·.  ·*·  `  ·  . • `· ·
#  · ` ·  *·  `  .+ ·  ` ·* •·`    ·  Dive! ·*  ` · +• .·` . *·    `·*.·  ·` •.`
# ·`·*.`  ·   *. ·   + https://adventofcode.com/2021/day/2 · ` `  · *.`•      · 
# `+··* `.  `··•     `+·.   ·`.*  · ·      * ` · ·.+  •·· `    *· .  `*  · +·  ·

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> list[tuple[str, int]]:
    res = []
    for line in s.splitlines():
        direction, val = line.strip().split()
        res.append((direction, int(val)))

    return res


direction_bases: dict[str, NDArray[np.int_]] = {
    'forward': np.array([1, 0], dtype=int),
    'down': np.array([0, 1], dtype=int),
    'up': np.array([0, -1], dtype=int)
}


def execute_directions(*directions: tuple[str, int], include_aim=False) -> NDArray[np.int_]:
    """Execute the input direction instructions, starting from the origin.
    include_aim specifies whether to include the aim value when navigating."""
    
    position = np.array([0, 0])
    aim = 0

    for direction, distance in directions:
        if include_aim:
            if direction == "up":
                aim -= distance
            elif direction == "down":
                aim += distance
            elif direction == "forward":
                position += np.array([distance, distance*aim], dtype=int)
            else:
                raise ValueError
            #
        else:
            vec = direction_bases[direction]
            position += distance * vec
    
    return position


def solve(data: str) -> tuple[int|str, ...]:
    direction_data = parse(data)

    position = execute_directions(*direction_data)
    star1 = position[0] * position[1]
    print(f"Solution to part 1: {star1}")

    position2 = execute_directions(*direction_data, include_aim=True)
    star2 = position2[0] * position2[1]
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 2
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
