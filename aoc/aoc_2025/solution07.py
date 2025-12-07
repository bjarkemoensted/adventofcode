# *` · · .+  ·    ·*`.·•  · ` +`· .  · +.·*•· `+ .· `  · +·* `.    +·.·* `·. * ·
# ·  +*  `. ·*.· +` + .· · + `  •. Laboratories +    . *·  . · ` +.    ·  `* · •
# ` · ` .·· *      ·   https://adventofcode.com/2025/day/7 ·` +·   · `  *.+·  .·
# .·` ·*·+ *` ·`·   · + . `  *·  *·   •`·+.·    · *`+*· `.  * ·+`·       · .* · 

import typing as t
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray

coordtype: t.TypeAlias = tuple[int, int]


@dataclass
class Beam:
    """Represents a beam. It keeps track of the points that have been visited
    by the beam in the past, and the wavefront of current points.
    Points in the wavefront have not yet been added to the set of visited points.
    Each of the attributes is represented by a counter which maps points to their multiplicities,
    meaning the number of distinct timeline in which each point has been reached."""

    history: Counter[coordtype] = field(default_factory=Counter)
    wavefront: Counter[coordtype] = field(default_factory=Counter)

    def copy(self) -> t.Self:
        return replace(self)

    def add_point_to_front(self, point: coordtype, multiplicity: int) -> t.Self:
        """Adds a new point with a given multiplicity to the wavefront."""
        self.wavefront[point] += multiplicity
        return self
    
    def add_point_to_history(self, point: coordtype, multiplicity: int) -> t.Self:
        """Adds a new point with a given multiplicity to the history."""
        self.history[point] += multiplicity
        return self

    def pop_wavefront(self) -> dict[coordtype, int]:
        """Adds the current wavefront to the history, and return it
        after clearing. When updating the state of a beam, this can be called to
        update history and get the current front, so the beam is ready for new points
        to be added to the front."""
        self.history += self.wavefront
        res = {k: v for k, v in self.wavefront.items()}
        self.wavefront.clear()
        return res

    def __getitem__(self, point: coordtype) -> int:
        """Look up the multiplicity of a point"""
        return self.history[point] + self.wavefront[point]
    
    @property
    def coords(self) -> set[coordtype]:
        return set(self.history.keys()) | set(self.wavefront.keys())

    def __contains__(self, x: coordtype) -> bool:
        # A point is contained in the beam if it has been visited in >0 timelines
        return self[x] > 0


class Symbol(str, Enum):
    START = "S"
    SPACE = "."
    SPLITTER = "^"
    BEAM = "|"


class Direction(Enum):
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line) for line in s.splitlines()])
    return res


def _shift_coord(point: coordtype, dir_: Direction) -> coordtype:
    """Shifts a point one step in the given direction"""
    i, j = point
    di, dj = dir_.value
    return i+di, j+dj


class Manifold:
    """Represents the Tachyon Manifold thingy.
    This is intended to be static, and is only responsible for updating beam states."""

    def __init__(self, layout: NDArray[np.str_]) -> None:
        self.layout = layout

    def coords_with_symbol(self, symbol: Symbol) -> set[coordtype]:
        res = {(i, j) for (i, j), char in np.ndenumerate(self.layout) if char == symbol}
        return res

    def in_bounds(self, *points: coordtype) -> bool:
        """Checks if one or more points are within the bounds of the layout.
        If multiple points are passed, returns True only if all are in bounds."""
        for p in points:
            if not all(0 <= coord < lim for coord, lim in zip(p, self.layout.shape)):
                return False
            #
        return True

    def evolve_beam(self, beam: Beam, n_steps=-1) -> Beam:
        """Takes some points, representing the current wavefront of a Tachyon beam.
        Returns a tuple consisting of the new wavefront, after each beam has
        travelled n_steps (which can be omitted to travel until the beam terminates)."""
        
        res = beam.copy()
        nits = 0

        while beam.wavefront:
            nits += 1
            if n_steps != -1 and nits > n_steps:
                break
            
            # Update beam and grab the current front
            front = res.pop_wavefront()

            for x, multiplicity in front.items():
                # Let the beam travel down a step. Abort if out of bounds
                xp = _shift_coord(x, Direction.DOWN)
                if not self.in_bounds(xp):
                    continue
                
                match self.layout[*xp]:
                    case Symbol.SPACE:
                        # If we hit free space, the new point goes to the front
                        res.add_point_to_front(xp, multiplicity)
                    case Symbol.SPLITTER:
                        # If it's a splitter, put it directly in the history, and check the 2 neighbor points
                        res.add_point_to_history(xp, multiplicity)
                        for dir_ in (Direction.LEFT, Direction.RIGHT):
                            # Add the neighbor point to the front unless out of bounds
                            x_split = _shift_coord(xp, dir_)
                            if self.in_bounds(x_split):
                                res.add_point_to_front(x_split, multiplicity)
                            #
                        #
                    case _:
                        raise RuntimeError
                    #
                #
            #
        
        return res

    def visualize_beam(self, beam: Beam) -> None:
        """Helper method for debugging. Prints the ASCII layout
        of the manifold, and the curremt state of a beam"""
        m = self.layout.copy()
        for i, j in beam.coords:
            if m[i, j] == Symbol.SPACE:
                m[i, j] = Symbol.BEAM.value
            #
        
        s = "\n".join(["".join(row) for row in m])
        print(s)
    #


def solve(data: str) -> tuple[int|str, ...]:
    layout = parse(data)
    manifold = Manifold(layout)

    start = manifold.coords_with_symbol(Symbol.START).pop()
    beam = Beam().add_point_to_front(start, multiplicity=1)
    beam = manifold.evolve_beam(beam)

    splitters = manifold.coords_with_symbol(Symbol.SPLITTER)
    star1 = sum(coord in beam for coord in splitters)
    print(f"Solution to part 1: {star1}")

    star2 = 1 + sum(beam[coord] for coord in splitters)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 7
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
