# `·.  +· `·`·.  * ·  • .·*    ·*`·   .     .··   +  . ··* *·  `.   +·`·  ·* `·.
# ·`*·· `. *·.·   ` .··*   ` ·    Spiral Memory *.·. •` ` · .·    · ·.*   . `·*·
#   ·+.`·*· * `.·  .   https://adventofcode.com/2017/day/3 `   ·  .  *·`·    .·*
#  · .*·  .· ` ·    ·*``·.. +  .·  ·*  ·.  •` .    · `·+ ·`.*    ·    .* ·`··*.`


import math


def parse(s: str):
    res = int(s)
    return res


def determine_coord(n):
    # Determine the largest odd number x such that x**2 <= n
    root = math.floor(n**0.5)
    lower = root - 1 if root % 2 == 0 else root

    # Determine which 'layer' in the spiral is the outermost in the contained square
    layer = (lower - 1) // 2
    # If exact root (e.g. n = 9, 25, 49, etc), the coordinate at the lower left corner
    if lower**2 == n:
        return layer, -layer

    # Otherwise, we're in the next layer out
    layer += 1
    circumference = (lower + 2)**2 - lower**2
    side_len = circumference // 4

    # Number of steps taken along the spiral from the lower left corner
    remainder = n - lower ** 2
    # Decompose into number of side lengths and remainder
    n_sides = remainder // side_len
    n_steps = remainder % side_len

    # Compute x and y coords
    x = layer - n_steps*(n_sides == 1) - (side_len - n_steps)*(n_sides == 3)
    y = -layer + n_steps*(n_sides == 0) - (side_len - n_steps)*(n_sides == 2)

    return x, y


def dist(coord: tuple):
    res = sum(map(abs, coord))
    return res


def _iterate_coords():
    """Iterates in a spiral out from the center (coordinate (0, 0))."""
    yield 0, 0
    layer = 0
    while True:
        layer += 1
        side_len = 1 + 2*layer
        coord = [layer, -layer]
        for i, shift in ((1, 1), (0, -1), (1, -1), (0, 1)):
            for _ in range(side_len - 1):
                coord[i] += shift
                yield tuple(coord)
            #
        #
    #


def _iterate_neighbors(coord):
    """Iterates over all 8 neighbors given an (x, y) coordinate tuple."""
    x, y = coord
    shifts = (-1, 0, +1)
    for dx in shifts:
        for dy in shifts:
            if not (dx == dy == 0):
                yield x + dx, y + dy
            #
        #
    #


def first_safety_check_val_greater_than(val: int) -> int:
    """Repeatedly fills out cells in a spiraling pattern with the sum of all pre-filled adjacent cells, starting
    with 1 at the origo. Returns the first value exceeding the input."""

    i = 0
    gen = _iterate_coords()
    squares = {next(gen): 1}
    for coord in gen:
        i += 1
        square = sum(squares.get(neighbor, 0) for neighbor in _iterate_neighbors(coord))

        if square > val:
            return square

        squares[coord] = square
    
    raise ValueError("No valid value found")


def solve(data: str) -> tuple[int|str, int|str]:
    n = parse(data)

    coord = determine_coord(n)
    star1 = dist(coord)
    print(f"Solution to part 1: {star1}")

    star2 = first_safety_check_val_greater_than(n)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
