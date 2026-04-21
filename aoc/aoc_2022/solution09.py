# ··. ·`` *   .· ··•  `   ·* · ·.·     .`·  +· `   ·· +  `    ·.·*`. +·` *·.·.·`
# ` ·+. ·`  ·*`  . ·`  . ·  +`  ·  Rope Bridge +·    · .·*     ·+`.  `*·.·  ` · 
# · ·.* +.`  · ·  . ·  https://adventofcode.com/2022/day/9 ·   • · ·.*  · .  ·`·
#  ``·`·.•    ·    ` · · .  `*·    ·`.    · `+· `    *·  .` ··    ·+ · *  `·•·.·

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> list[tuple[str, int]]:
    instructions = []
    for line in s.split("\n"):
        a, b = line.split(" ")
        instructions.append((a, int(b)))

    return instructions



def get_directions() -> dict[str, NDArray[np.int_]]:
    """Provides a mapping from a direction (LRDU) to a corresponding unit vector."""
    d = dict(
        L=np.array([-1, 0]),
        R=np.array([1, 0]),
        U=np.array([0, 1]),
        D=np.array([0, -1])
    )
    return d


def compute_tail_move_direction(head: NDArray[np.int_], tail: NDArray[np.int_]) -> NDArray[np.int_]:
    """Computes the direction a 'tail' point in the rope will be pulled by another 'head' point."""
    delta = head - tail
    touch = all(val <= 1 for val in np.abs(delta))
    # If the points touch, do not move the tail
    if touch:
        return np.array([0, 0])

    # Otherwise, move the tail a maximum of one unit in both the x and y directions
    return np.clip(delta, -1, 1)


def simulate_rope(instructions: list[tuple[str, int]], length: int) -> list[tuple[int, int]]:
    # List maintaining the points visited by the tail of the rope
    history = [(0, 0)]

    # The positions of all knots on the rope (head plus n - 1 tail knots)
    rope = [np.array([0, 0]) for _ in range(length)]

    direction_dict = get_directions()

    for direction, n_steps in instructions:
        step = direction_dict[direction]  # get the direction unit vector, e.g. 'L' -> (-1, 0)
        for _ in range(n_steps):
            for i, pos in enumerate(rope):
                is_head = i == 0
                is_tail = i == length - 1
                if is_head:
                    # If moving the head knot, just move it in the specified direction
                    new_position = pos + step
                else:
                    # If moving a tail knot, use the next knot to determine update position
                    next_point = rope[i-1]
                    move_vector = compute_tail_move_direction(next_point, pos)
                    new_position = pos + move_vector
                rope[i] = new_position
                if is_tail:
                    history.append(tuple(new_position))
                #
            #
        #

    return history


def solve(data: str) -> tuple[int|str, ...]:
    instructions = parse(data)

    history = simulate_rope(instructions, length=2)
    star1 = len(set(history))
    print(f"Solution to part 1: {star1}")

    history2 = simulate_rope(instructions, length=10)
    star2 = len(set(history2))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 9
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
