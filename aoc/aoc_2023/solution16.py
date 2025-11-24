# ·· .    ···`.*· ·`.`   ·· ·  •·  `.  .·` * ·· •.  ·   ·`  ·  *·  ` ·`.  . *·`·
# .*`·  ·    *·`.·  •*`   .   The Floor Will Be Lava ·  . ·   · `* .·  `   ·`··.
# `.·*·  · `   ·    ·. https://adventofcode.com/2023/day/16  · .+ ·• `.·`··   .·
# *·.` ·.`   · .+` ·  ··  `.*· ·    ·.*`  .·  ·     + ·.`··•   ·.`.*  ·*·` ·· · 

from functools import cache
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

coordtype: TypeAlias = tuple[int, int]


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line) for line in s.split("\n")])
    return res


up = (-1, 0)
right = (0, 1)
down = (1, 0)
left = (0, -1)


class Grid:
    def __init__(self, map_: NDArray[np.str_]):
        self.map_ = map_.copy()

    @cache
    def update_direction(self, crd: coordtype, dir_: coordtype) -> tuple[coordtype, ...]:
        """Returns the direction(s) of light beam(s) after a beam has visited input coordinate."""
        
        char = self.map_[crd]
        if char == ".":
            return (dir_,)
        elif char == "|":
            if dir_ in (up, down):
                return (dir_,)
            else:
                return (up, down)
        elif char == "-":
            if dir_ in (left, right):
                return (dir_,)
            else:
                return (right, left)
        elif char == r"/":
            return ({right: up, up: right, left: down, down: left}[dir_],)
        elif char == "\\":
            return ({right: down, down: right, left: up, up: left}[dir_],)
        else:
            raise ValueError(f"Can't update with char: {char}")
        #
    
    @cache
    def step(self, crd: coordtype, dir_: coordtype) -> coordtype|None:
        """Updates a coordinate given a direction. Returns None if taking the step falls off the map"""
        newcrd = tuple(a + b for a, b in zip(crd, dir_))
        if not all (0 <= x < lim for x, lim in zip(newcrd, self.map_.shape)):
            return None
        
        assert len(newcrd) == 2
        return newcrd

    @cache
    def shine(self, initial_wavefront: tuple[coordtype, coordtype]|None=None) -> int:
        """Computes the final number of energized tiles given an initial wavefront (coord + direction tuple)."""

        if initial_wavefront is None:
            initial_wavefront = ((0, 0), (0, 1))

        energized = set([])  # Keep track of which tiles are energized
        wavefronts = {initial_wavefront}  # Keep track of the wavefronts we have to update
        wavefront_history: set[tuple[coordtype, coordtype]] = set({})  # Keep track of wavefronts we've seen before

        # Keep updating as long as we're encountering previously unseen wavefronts
        while wavefronts - wavefront_history:
            new_wavefronts = set([])

            for wavefront in wavefronts:
                wavefront_history.add(wavefront)
                crd, dir_ = wavefront
                energized.add(crd)
                dirs = self.update_direction(crd, dir_)
                for newdir in dirs:
                    newcrd = self.step(crd, newdir)
                    if newcrd is not None:
                        new_wavefront = (newcrd, newdir)
                        if new_wavefront not in wavefront_history:
                            new_wavefronts.add(new_wavefront)
                #
            wavefronts = new_wavefronts

        n_energized = len(energized)
        return n_energized
    #

    def find_highest_energization(self):
        """Just try all sites along the edge of the map with all light beams going inwards."""

        rows, cols = self.map_.shape
        # Define all possible initial wavefronts whining inwards from the edges
        initial_wavefronts = []
        for i in range(rows):
            initial_wavefronts.append(((i, 0), right))  # left shining right
            initial_wavefronts.append(((i, cols - 1), left))  # right shining left
        for j in range(cols):
            initial_wavefronts.append(((0, j), down))  # up shining down
            initial_wavefronts.append(((rows - 1, j), up))  # down shining up

        # Find the max number of energized tiles
        res = max(self.shine(wf) for wf in initial_wavefronts)
        return res


def solve(data: str) -> tuple[int|str, ...]:
    map_ = parse(data)
    grid = Grid(map_)

    star1 = grid.shine()
    print(f"Solution to part 1: {star1}")

    star2 = grid.find_highest_energization()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 16
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
