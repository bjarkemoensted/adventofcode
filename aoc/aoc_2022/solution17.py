# `··*.  · `     ·`.       `*··  .+`·  .··  `·•· . • +.` ·     ·   `·   .·`  ··*
# ·*·``+   ··`·  . `·.*  ··+ `   Pyroclastic Flow  · .·*      ·   · *`` ·+. ·`.·
# · •.··.   *   ·`· .· https://adventofcode.com/2022/day/17 ··*  ·.·  •· `   .`·
#  .`· *·  .`·.  * ·`•··  . ` .·  `·· *    `  .`·+· .*   `··+.  ·   · .+  · * ·`

from typing import Hashable, Iterable, Iterator

import numpy as np
from numpy.typing import NDArray

type arrtype = NDArray[np.bool_]
type coordtype = NDArray[np.int_]
type keytype = tuple[int, int, int, Hashable]


_ON = "#"
_OFF = "."
_FALLING = "@"
map_ = {_ON: True, _OFF: False}
map_inv = {v: k for k, v in map_.items()}

rock_shapes = """####

.#.
###
.#.

..#
..#
###

#
#
#
#

##
##"""


def make_rocks() -> Iterator[coordtype]:
    """Iterate over the coordinates (ij) of the blocks, in order."""
    for part in rock_shapes.split("\n\n"):
        arr = np.array([list(line) for line in part.splitlines()])
        # Flip the shape so greater i -> higher
        flipped = np.flip(arr, axis=0)
        coords = np.argwhere(flipped == _ON)
        yield coords
    #


dirmap: dict[str, coordtype] = {
    ">": np.array([0, 1]),
    "<": np.array([0, -1]),
}


def parse(s: str) -> list[str]:
    return list(s)


class Tetris:
    """Represents the Tetris-like scenario with rocks falling down a narrow cavern and jets
    shifting their position left and right.
    To simplify array expansions and indicing, coordinates i, j here mean the y and x coordinates,
    meaning rocks with greater values of i are located higher."""

    # Space left of, and below, new rocks
    SPACE_LEFT = 2
    SPACE_UP = 3

    def __init__(self, directions: Iterable[str], width=7) -> None:
        self.dirs = tuple(dirmap[dir_].copy() for dir_ in directions)
        self.rocks = tuple(make_rocks())
        self.dir_ind = 0
        self.rock_ind = 0
        self.n_rocks_fallen = 0
    
        self.width = width
        self.arr = np.full(shape=(1, self.width), fill_value=False, dtype=bool)

        # Bottom row - the lowest y-values we consider, starting above a full blocked line
        self.bottom = 0
        # Current peak (start at minus one to indicate there is none)
        self.peak = 0

    def _expand_array(self) -> None:
        """Doubles the array storing the fallen rocks"""
        space = np.zeros_like(self.arr).astype(bool)
        self.arr = np.vstack([self.arr, space])

    def _get_next_dir(self) -> coordtype:
        """Get the next direction of the gas jet"""
        res = self.dirs[self.dir_ind].copy()
        self.dir_ind = (self.dir_ind + 1) % len(self.dirs)
        return res
    
    def _get_next_rock(self) -> coordtype:
        """Get the next falling rock"""
        res = self.rocks[self.rock_ind].copy()
        self.rock_ind = (self.rock_ind + 1) % len(self.rocks)
        return res

    def spawn_rock(self) -> coordtype:
        """Set up the next rock which is going to fall."""
        rock = self._get_next_rock()

        # If there's not enough space for the rock, expand the data array
        while rock[:, 0].max() + self.SPACE_UP + self.peak >= self.arr.shape[0]:
            self._expand_array()

        # Shift the rock to appear at the correct position
        lowest, leftmost = rock.min(axis=0)
        shift = np.array([self.peak + self.SPACE_UP - lowest, self.SPACE_LEFT - leftmost])
        rock += shift

        # Check bounds and position criteria
        assert np.all(0 <= rock[:, 1]) and np.all(rock[:, 1] < self.width)
        assert not np.any(rock[:, 0] >= self.arr.shape[0])
        assert rock[:, 1].min() == self.SPACE_LEFT
        assert rock[:, 0].min() - self.peak == self.SPACE_UP

        return rock
    
    def collision(self, rock: coordtype) -> bool:
        """Checks if a rock with the input coordinates would collide with other rocks or the cavern walls/floor"""

        # Collision if the rock went out of bounds (collides with wall or floor)
        if np.any(rock < 0):
            return True
        out_of_bounds = rock >= np.array(self.arr.shape)
        if np.any(out_of_bounds[:, 0]):
            raise RuntimeError  # Shouldn't go out of bound upwards
        elif np.any(out_of_bounds[:, 1]):
            return True

        # Check for collision with other rocks
        hit_rock = np.any(self.arr[rock[:, 0], rock[:, 1]]).item()
        return hit_rock

    def _register_rock_position(self, rock: coordtype) -> None:
        """Processes when a rock comes to a rest in the cavern. Updates data array, peak, and bottom."""

        ymin, ymax = np.unique(rock[:, 0], sorted=True)[[0, -1]]

        rock_peak = int(ymax) + 1
        if rock_peak > self.peak:
            self.peak = rock_peak
        
        if self.collision(rock):
            raise RuntimeError
        
        self.arr[rock[:, 0], rock[:, 1]] = True

        for y in range(ymax+1, ymin-1, -1):
            low = y-2
            if low < 0:
                break

            band = self.arr[low:y, :]
            blocked = np.all(np.any(band, axis=0))
            if blocked:
                self.bottom = y - 1
                return None
            #
        #

    def drop_rock(self) -> None:
        """Spawns a new rock and lets it fall"""
        
        rock = self.spawn_rock()

        while True:
            # The hot air jet moves the rock to the side (unless it collides with something)
            dir_ = self._get_next_dir()
            new_pos = rock + dir_
            if not self.collision(new_pos):
                rock = new_pos
            
            # Rock falls down one step. Stop falling if it hits something
            new_pos = rock + np.array([-1, 0])
            if self.collision(new_pos):
                self._register_rock_position(rock)
                return None
            else:
                rock = new_pos
            #
        #

    def display(self, block: coordtype|None=None) -> None:
        """Helper method for visualizing a state"""
        low = max(0, self.bottom)
        high = (self.peak if block is None else max(self.peak, block[:, 0].max())) + 1
        a = np.vectorize(map_inv.get)(self.arr[low: high, :])
        if block is not None:
            a[block[:, 0], block[:, 1]] = _FALLING
        
        s = "\n".join(["".join(row) for row in np.flip(a, axis=0)])
        print(s, end="\n\n")

    def compute_key(self) -> keytype:
        """Computes a hashable key, representing the current state"""
        
        # Only consider reachable coordinates
        active_region = self.arr[self.bottom: self.peak+1]
        region_size, _ = active_region.shape

        # Get the bytes representation of the active region
        array_state = np.ascontiguousarray(active_region).tobytes()
        key = (self.rock_ind, self.dir_ind, region_size, array_state)

        return key


def compute_height(directions: list[str], n_rocks=2022) -> int:
    """Computes the height of the tower resulting from dropping n rocks, with the specified wind
    directions."""

    first_seen: dict[keytype, int] = dict()
    heights: list[int] = []

    tetris = Tetris(directions)

    for i in range(n_rocks):
        # Drop one more rock
        tetris.drop_rock()
        # Check if the resulting state has been encountered before
        key = tetris.compute_key()
        recurrence = key in first_seen
        
        if recurrence:
            # Determine the number of steps in the recurrence cycle
            ind_last = first_seen[key]
            cycle_duration = i - ind_last

            # Compute growth during one cycle
            current_height = tetris.peak
            old_height = heights[ind_last]
            grown_by = current_height - old_height

            # Number of loops and remainder
            n_rocks_left = n_rocks - i - 1
            n_cycles, remainder = divmod(n_rocks_left, cycle_duration)

            # Total growth from the loops and remaining rocks after loops
            cycles_total_growth = n_cycles * grown_by
            remainder_growth = heights[ind_last+remainder] - heights[ind_last]
            res = remainder_growth + current_height + cycles_total_growth

            return res
        
        # If no recurrence, store the key so we can detect recurrences to this state
        first_seen[key] = i
        heights.append(tetris.peak)
    
    res = tetris.peak
    return res


def solve(data: str) -> tuple[int|str, ...]:
    directions = parse(data)
    star1 = compute_height(directions)
    print(f"Solution to part 1: {star1}")

    n = 1_000_000_000_000
    star2 = compute_height(directions, n_rocks=n)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 17
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
