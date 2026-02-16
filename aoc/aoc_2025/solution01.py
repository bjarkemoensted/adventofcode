#  `·*`.·    · .`      ·` ·*.`  ` ·  . · * `·` + ·  . · ·   ``*· .·* `  .`·+.··`
# .· ··*   `  ·* .·  .·`.  `·+·  Secret Entrance *  `· .  ·.  ·* `  .  ·*·`.  .·
# · `.* ·`·+   · `  *` https://adventofcode.com/2025/day/1  ·   `·* ·  .· +·  · 
# `·+    . ·   .·  ` ·*. · ·•   ·+` .·*` ·`   .·   ·* ` · .  · `  ·`*.·  .` · +.

from typing import Callable, Literal


def parse(s: str) -> list[tuple[Literal[+1, -1], int]]:
    """Parses into tuples of (sign, distance).
    sign is +/- 1, indicating the direction of the rotation."""

    map_: dict[str, Literal[+1, -1]] = {"R": +1, "L": -1}
    res: list[tuple[Literal[+1, -1], int]] = []

    for line in s.splitlines():
        sign = map_[line[0]]
        number = int(line[1:])
        res.append((sign, number))

    return res


def count_zero_pos(magnitude: int, mod: int) -> int:
    """Counts whether the dial ends on zero (returns 1 if yes, otherwise 0)"""
    return int(magnitude % mod == 0)


def count_zero_passes(magnitude: int, mod: int) -> int:
    """Counts the number of times the dial passes 0.
    The magnitude must be a positive position, before taking the modulo of the dial's size."""

    if magnitude < 0:
        raise ValueError("Magnitude cannot be negative")
    return magnitude // mod


def _reflect(sign: Literal[+1, -1], x: int, mod: int) -> int:
    """Reflect the input number around 0, with the input modulo.
    This determines how far from 'mod' the number x is, in the direction given by the sign.
    Examples:
    -1, 10, 100 -> 90,
    +1, 42, 100 -> 42"""

    res = (sign*x) % mod
    return res


def analyze_rotations(
        rotations: list[tuple[Literal[-1, +1], int]],
        accumulator: Callable[[int, int], int],
        start=50,
        size=100) -> int:
    """Perform the input rotations, with a dial of the specified size and starting position.
    To avoid issues with left rotations, reflect around 0 before adding a new rotation, and undo the reflection
    afterwards. Between the two reflections, the accumulator callable is used to count the number of events of
    interest, based on the magnitude (the dial's position before taking the modulo and unreflecting),
    and the dial size."""

    res = 0
    pos = start

    for sign, number in rotations:
        # Reflect and handle accumulation
        magnitude = _reflect(sign, pos, size) + number
        res += accumulator(magnitude, size)
        
        # Determine dial position, then unreflect
        reflected_pos = magnitude % size
        pos = _reflect(sign, reflected_pos, size)
    
    return res


def solve(data: str) -> tuple[int|str, ...]:
    rotations = parse(data)

    star1 = analyze_rotations(rotations, accumulator=count_zero_pos)
    print(f"Solution to part 1: {star1}")

    star2 = analyze_rotations(rotations, accumulator=count_zero_passes)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 1
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
