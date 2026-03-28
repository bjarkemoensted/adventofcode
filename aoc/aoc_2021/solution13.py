# ·•` ·  `   · * · ·* · .  `  * ·`   · .* ·* ` ·      *··    ··`+.·*`.  ·  ·` ·*
# *·· `· ·+ · *   . ·  *`   ·  Transparent Origami `  .  ·      ·`* ·   .·`+·  ·
#  · `  *.·· `    ·+ · https://adventofcode.com/2021/day/13 ·*`  ··  • ·`*·.  ·.
# ·`·  *· .` *·   ` ·* ·  ·  ·  .+ ·*` · . ·       ·•··`*    ` · * · ·    . *·`·

import numpy as np
from aococr import aococr
from numpy.typing import NDArray

PIXEL_ON = "#"
PIXEL_OFF = "."


def parse(s: str) -> tuple[NDArray[np.int_], list[tuple[str, int]]]:
    a, b = s.split("\n\n")
    
    xy_coords = np.array([[int(part) for part in line.split(",")] for line in a.splitlines()])
    inds = np.flip(xy_coords, axis=1)
    shape = tuple(inds.max(axis=0) + 1)
    
    M = np.zeros(shape=shape, dtype=int)
    M[inds[:, 0], inds[:, 1]] = 1

    folding_instructions = []
    for line in b.split("\n"):
        direction, val_str = line.split("fold along ")[1].split("=")
        val = int(val_str)
        folding_instructions.append((direction, val))
    
    return M, folding_instructions


def fold(m: NDArray[np.int_], axis: str, coord: int) -> NDArray[np.int_]:
    """Returns a folded paper. For example, pass m=some matrix, axis='y', coord + 42.
    This will return the input matrix folded around y=42. The lower part is flipped and
    added to the upper part, like when folding a paper."""

    # Copy data so we don't mess anything up
    m = m.copy()

    # If we're folding around x=something, partition into a left and right half
    if axis == 'x':
        ma, mb = m[:, :coord], m[:, coord+1:]
        flipaxis = 1
    # Similarly, partition into upper and lower when flipping around y=...
    elif axis == "y":
        ma, mb = m[:coord, :], m[coord+1:, :]
        flipaxis = 0
    else:
        raise ValueError

    # The 'halves' don't always line up exactly, so zero-pad the latter half so the shapes match.
    b2 = np.zeros_like(ma)
    b2[:mb.shape[0], :mb.shape[1]] = mb
    mb = b2

    if ma.shape != mb.shape:
        raise ValueError(f"Matrix a shape ({ma.shape}) != matrix b shape ({mb.shape})")

    # Flip, add and return
    mb = np.flip(mb, axis=flipaxis)
    res = ma + mb
    return res


def solve(data: str) -> tuple[int|str, ...]:
    M, instructions = parse(data)

    # Do the first folding instruction, and count the overlaps
    M_first_fold = fold(M, *instructions[0])
    star1 = (M_first_fold > 0).sum()
    print(f"Solution to part 1: {star1}")

    # Do all the folds
    for axis, coord in instructions:
        M = fold(M, axis, coord)
    
    # Make results more readable by converting zeroes to '.' and nonzeroes to '#'
    M_display = np.full(shape=M.shape, fill_value=PIXEL_OFF, dtype=str)
    M_display[np.where(M > 0)] = PIXEL_ON
    
    # Parse the characters appearing on the display
    star2 = aococr(M_display)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 13
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
