# ·.`*   · *.·` ·     •··      ·.·*•`. ` ·+· *   ·.·`  ·   *·. ·  ·`  +   *·.`··
# ·`*` ·   ·*` •  . · ·`   .`·*   Seating System .+·    •  ·.*  ·· ·   .·  .  *`
# .*··•` ··.`*     · + https://adventofcode.com/2020/day/11 *`·. *  .·   •·  · .
# *·. ` . * · .·  ·* `   ·*·.     `·  • ·   .`.··  `·* •.·      `. .·* · ·`*  .·

import itertools
import typing as t

import numba
import numpy as np
from numpy.typing import NDArray

coordtype: t.TypeAlias = tuple[int, int]

_seat = "L"
_occupied = "#"


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line.strip()) for line in s.splitlines()])
    return res


@numba.njit(cache=True)
def update_state(state: NDArray[np.int_], neighborhood: NDArray[np.int_], crowd_thres: int) -> NDArray[np.int_]:
    """Updates the input state in a simulation.
    state is a 2D matrix with 1 representing occupied seats and 0 representing vacant seats
    or floor sites.
    neighboorhood contains for each i, j site the up to 8 seats considered neighbors of i, j, with -1
    denoting no seat.
    crowd_thres is the threshold for when a site is 'crowded' and an occupied seat will change into
    a vacant seat."""
    
    res = state.copy()

    for (i, j), occupied in np.ndenumerate(state):
        # If the sites has no neighborhood at all, it's a floor site
        if np.all(neighborhood[i, j][0] == -1):
            continue
        
        # Count the number of occupied neighbors
        n_adj = 0
        for ni, nj in neighborhood[i, j]:
            if ni == nj == -1:
                break
            n_adj += state[ni, nj]

        # Update the site according to the rules
        if occupied == 1:
            # Occupied seats turn vacant if their neighborhood is crowded
            if n_adj >= crowd_thres:
                res[i, j] = 0
            #
        # Vacant seats are only occupied if their entire neighborhood is free
        elif n_adj == 0:
            res[i, j] = 1
        #

    return res


class Simulation:
    def __init__(self, map_: NDArray[np.str_], line_of_sight=False) -> None:
        """map_ is the ASCII map of the seat layout.
        line_of_sight denotes whether seats consider not just their immediate vicinity,
        but the first seat along any of the 8 directions."""

        self.map_ = map_
        self.shape = self.map_.shape
        self.line_of_sight = line_of_sight
        self.crowded_threshold = 5 if self.line_of_sight else 4

        self.seats = {(i, j) for (i, j), char in np.ndenumerate(map_) if char == _seat}
        self.neighborhoods = self._determine_neighborhoods()
    
    def _in_bounds(self, x: coordtype) -> bool:
        return all(0 <= coord < lim for coord, lim in zip(x, self.shape, strict=True))

    def _get_neighbor_sites(self, i: int, j: int) -> t.Iterator[coordtype]:
        """Generates the sites that count as neighbors of (i, j)."""

        # DEfine the 8 directions
        steps = (-1, 0, +1)
        directions = ((di, dj) for di in steps for dj in steps if not (di == dj == 0))

        for di, dj in directions:
            # Iterator for all points given by (i, j) + n*(di, dj), where n is a positive int
            line = ((i + n*di, j + n*dj) for n in itertools.count(start=1, step=1))
            # Continue along the line while its points are within bounds
            for p in itertools.takewhile(self._in_bounds, line):
                # Stop if we encounter a seat
                if p in self.seats:
                    yield p
                    break
                # Or if we're not using line-of-sight (so we only look at immediate neighbors)
                if not self.line_of_sight:
                    break
                #
            #
        
    def _determine_neighborhoods(self) -> NDArray[np.int_]:
        """Determines for each seat the coordinates of the seats that might influence the seat.
        For non-line-of-sight simulations, this finds the (up to) 8 adjacent sites.
        For line-of-sights, finds the first seat (if any) in each of the 8 directions.
        The result is represented in a 4D numpy array such that the rows of M[i, j] contains
        the neighbors for the seat at i, j. Rows of (-1, -1) are used to denote no neighbor."""

        n_neighbors_max = 8
        dim = len(self.shape)
        res = -np.ones((*self.shape, n_neighbors_max, dim), dtype=int)

        for i0, j0 in np.ndindex(self.shape):
            if (i0, j0) not in self.seats:
                continue
            for n_ind, neighbor in enumerate(self._get_neighbor_sites(i0, j0)):
                for nind, ncoord in enumerate(neighbor):
                    res[i0, j0][n_ind][nind] = ncoord
                #
            #

        return res

    def visualize_state(self, state: NDArray[np.int_]) -> str:
        """Helper method for visualizing a state"""
        viz = self.map_.copy()
        for (i, j), val in np.ndenumerate(state):
            if val == 1:
                viz[i, j] = _occupied

        lines = ["".join(row) for row in viz]
        res = "\n".join(lines)
        return res

    def run(self, state: NDArray[np.int_], n_steps=-1) -> NDArray[np.int_]:
        """Runs a simulation. n_steps is optional - if left at -1, will continue
        until the simulation stabilizes (its state doesn't change).
        Returns the final state, which can be summed to obtain the number of occupied
        seats"""
        
        for n in itertools.count(start=1, step=1):
            next_state = update_state(state, neighborhood=self.neighborhoods, crowd_thres=self.crowded_threshold)
            stabilized = np.all(state == next_state)
            state = next_state
            
            if stabilized or (n_steps != -1 and n >= n_steps):
                break
            #
        
        return state
        
def solve(data: str) -> tuple[int|str, ...]:
    map_ = parse(data)
    x0 = np.zeros(map_.shape, dtype=int)

    star1 = Simulation(map_).run(x0).sum()
    print(f"Solution to part 1: {star1}")

    star2 = Simulation(map_, line_of_sight=True).run(x0).sum()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 11
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
