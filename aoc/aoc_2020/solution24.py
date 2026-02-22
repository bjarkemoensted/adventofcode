# *·`  .    `.  ·.` *+··`·.    ·` * ·.*.· `•.   · ` · +·.   .` .*·  · `.   ·.•·`
# + ··`·  ·.**`.  ·   `+  *·   .   Lobby Layout   *·`.·  + .  `*·   .`*   .   ·.
# ·`· ·*`..•    ·    . https://adventofcode.com/2020/day/24 `.·+. · *.   · .`·+·
# .+. * ·  ` ·.    `·  .  ·* ·.` + .·  ·` ·.*.·` +   · . ` ·  .·    ·*·` *` · .+

import typing as t
from collections import Counter

import numpy as np
from numpy.typing import NDArray

postype: t.TypeAlias = tuple[int, int]
dirtype: t.TypeAlias = t.Literal["e", "w", "se", "ne", "sw", "nw"]
valid_dirs = set(t.get_args(dirtype))

dtype = np.int32  # numpy type for storing coordinates


def isdir(s) -> t.TypeGuard[dirtype]:
    return s in valid_dirs


# Directions scaled by factor of 2 (so directions like NE can also be integers)
direction_tuples: tuple[tuple[dirtype, postype], ...] = (
    ('e', (2, 0)),
    ('w', (-2, 0)),
    ('ne', (1, 2)),
    ('nw', (-1, 2)),
    ('se', (1, -2)),
    ('sw', (-1, -2))
)

dirs: dict[dirtype, NDArray[np.int_]] = {
    dir_: np.array(vec, dtype=dtype) for dir_, vec in direction_tuples
}


def parse(s: str) -> list[list[dirtype]]:
    res: list[list[dirtype]] = []
    lens = sorted(set(map(len, valid_dirs)))
    for line in s.splitlines():
        steps: list[dirtype] = []
        i = 0
        while i < len(line):
            for len_ in lens:
                step = line[i:i+len_]
                if isdir(step):
                    steps.append(step)
                    i += len_
                    break
                #
            #
        res.append(steps)
    return res


def position_from_steps(steps: list[dirtype]) -> postype:
    """Compute the final position, given a list of steps, summing the direction vector of each"""
    m = np.stack([dirs[step] for step in steps], dtype=dtype)
    x, y = m.sum(axis=0)
    return x, y


def count_coordinates(pos: NDArray[np.int_]) -> dict[postype, int]:
    """Takes an array where each row represents a point.
    Returns a dict mapping each coordinate ((x, y) format) to the number of times
    it occurs."""

    # Sort by both coordinates. After this, rows with same x/y values are grouped together
    pos = pos[np.lexsort((pos[:, 0], pos[:, 1]), axis=-1)]
    # Determine which rows differ in any column from previous row. Pad with 0 and len(arr)
    val_changes = np.any(pos[1:] != pos[:-1], axis=1)
    change_inds = np.empty(sum(val_changes) + 2, dtype=int)
    change_inds[0] = 0
    change_inds[-1] = len(pos)
    change_inds[1:-1] = np.where(val_changes)[0] + 1

    # The counts are the differences between consecutive indices where a change occurs
    unique_coords = pos[change_inds[:-1]]
    counts = np.diff(change_inds)

    # Return each coordinate and their count
    res = {coord: n for coord, n in zip(map(tuple, unique_coords), counts)}
    return res


def flip_simulation(black_tiles: t.Sequence[postype], n_iterations: int=1) -> list[postype]:
    """Simulates the changing tiles.
    black_tiles: The initial positions of the black tiles.
    n_iterations: The number of iterations to simulate"""

    black = set(black_tiles)
    all_dirs = np.stack([dir_ for _, dir_ in sorted(dirs.items())], dtype=dtype)

    for _ in range(n_iterations):
        # Compute a large array with all >0 number of black tile neighbors
        pos = np.stack([np.array(pos) for pos in sorted(black)], dtype=dtype)
        neighbors = (pos[:, None, :] + all_dirs[None, :, :]).reshape(-1, pos.shape[1])
        
        # Count the number of each coordinate
        counts = count_coordinates(neighbors)

        # Update tiles according to the rules
        black_pos = {(x, y) for x, y in pos}
        flip_to_white = {p for p in black_pos if counts.get(p, 0) == 0 or counts[p] > 2}
        flip_to_black = {p for p, n in counts.items() if n == 2 and p not in black_pos}
        black = (black - flip_to_white) | flip_to_black

    res = sorted(black)
    return res


def solve(data: str) -> tuple[int|str, ...]:
    steplists = parse(data)

    tile_counts = Counter(map(position_from_steps, steplists))
    black_tiles = [pos for pos, n in tile_counts.items() if n % 2 == 1]
    star1 = len(black_tiles)
    print(f"Solution to part 1: {star1}")

    black_after_flipping = flip_simulation(black_tiles=black_tiles, n_iterations=100)
    star2 = len(black_after_flipping)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 24
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
