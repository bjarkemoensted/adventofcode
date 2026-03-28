# .+`·· ·*`   .   `·   ` · •`·  +  ·· * .· *  ·  . · ` ·+   ·  · ` ·*. +.·*. ` ·
# · · *·`. ·`•*  ·  *·  .*··  ` ·  Smoke Basin  ·`*.··  `* ·  +  ·   `·  *.`·. ·
# ·*.·` .·*  ·   *     https://adventofcode.com/2021/day/9 *  ·`*..*   · `·*`· `
# *·*`·   .*· ··*` `·   `   ·    •*. ··.   ` · *`*· . ·+· `  *. · ·`   •·. ·* `.

from functools import cache, reduce
from operator import mul

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> NDArray[np.int_]:
    M = np.array([[int(s) for s in line.strip()] for line in s.splitlines()], dtype=int)
    return M


@cache
def get_neighborhood_coords(shape: tuple[int, int], i: int, j: int) -> list[tuple[int, int]]:
    """Takes a matrix (2d np array) shape and returns a list of coordinates of neighbors.
    Returns 4 points (E, W, S, N), unless input is at the edge/corner."""

    cands = ((i+1, j), (i-1, j), (i, j+1), (i, j-1))

    # Keep those that aren't out of bounds
    res = [(ip, jp) for ip, jp in cands if all(0 <= c < lim for lim, c in zip(shape, (ip, jp), strict=True))]
    return res


def get_neighborhood_values(M: NDArray[np.int_], i: int, j: int) -> list[int]:
    """Returns a list of the matrix elements around the input point"""
    return [M[a, b] for a, b in get_neighborhood_coords(M.shape, i, j)]


def is_low_point(M: NDArray[np.int_], i: int, j: int) -> bool:
    """Determine whether i,j is a 'low point', i.e. m[i,j] has a lower value than all its neighbors."""
    neighbors = get_neighborhood_values(M, i, j)
    val = M[i, j]
    return all(val < neighbor for neighbor in neighbors)


def find_unassigned_points(
        M: NDArray[np.int_],
        unassigned_coords: set[tuple[int, int]],
        target_value: int
    ) -> list[tuple[int, int]]:
    """Searchers the matrix for a target value, e.g. 0, which is in the set of unassigned crds"""
    return [coords for coords in unassigned_coords if M[coords] == target_value]


def find_basin_from_low_point(M: NDArray[np.int_], i: int, j: int) -> set[tuple[int, int]]:
    """Starts from the lowest point in a bassin. Repeatedly adds surrounding higher points,
    until no higher neighbors with height < 9 exist"""
    
    basin = set([])
    new_addition = {(i, j)}
    # As long as we've just added new points, check their neighbors for even higher points
    while new_addition:
        basin.update(new_addition)
        new_seeds = {tup for tup in new_addition}
        new_addition = set([])
        # Check the neighborhood of newly added points
        for a, b in new_seeds:
            seed_val = M[a, b]
            # If any neighbor is higher up and not already in bassin, add it
            for neighbor in get_neighborhood_coords(M.shape, a, b):
                i_n, j_n = neighbor
                neighbor_val = M[i_n, j_n]
                neighbor_can_flow_down = seed_val < neighbor_val < 9
                if neighbor_can_flow_down and neighbor not in basin:
                    new_addition.add(neighbor)
                #
            #
        #
    return basin


def sum_risk_scores(M: NDArray[np.int_]) -> int:
    low_points = []
    for i, j in np.ndindex(M.shape):
        if is_low_point(M, i, j):
            low_points.append((i, j))
        #

    # Find the sum of risk scores (1 + height of each low points)
    res = sum(M[i, j].item() + 1 for i, j in low_points)
    return res


def compute_basin_sizes(M: NDArray[np.int_]) -> list[int]:
    """Returns a list of the sizes of each basin"""
    # Coordinates that have not yet been assigned to a bassin
    unassigned_coords = {(i, j) for (i, j), val in np.ndenumerate(M) if val < 9}

    # Result dict - maps each bassin low point to the coordinates contained in the bassin
    low_point2basin = {}

    # Iterate over the heights of possible bassin low points (0 through 8)
    for target_value in range(9):
        low_points = find_unassigned_points(M, unassigned_coords, target_value)
        # For each low point, find the corresponding bassin
        for low_point in low_points:
            i, j = low_point
            bassin = find_basin_from_low_point(M, i, j)
            # Add to results dict, and remove bassin from the pool of coordinates not assigned to a bassin
            low_point2basin[low_point] = bassin
            unassigned_coords -= bassin
        #
    
    res = [len(bassin) for bassin in low_point2basin.values()]
    return res


def solve(data: str) -> tuple[int|str, ...]:
    M = parse(data)
    star1 = sum_risk_scores(M)
    print(f"Solution to part 1: {star1}")

    # Find the product of the sizes of the three largest bassins
    basin_sizes = compute_basin_sizes(M)
    star2 = reduce(mul, sorted(basin_sizes, reverse=True)[:3])
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 9
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
