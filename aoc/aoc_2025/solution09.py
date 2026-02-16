# ··`.`*• ··  * .· `. *   `·  .·  · ` *·.    ·` .+  ·  ·*`  ·   `.·`* .   .··*  
# `*  ·. ·  `  · *`. ·  +·*  .·   Movie Theater  · `  .` · *.   ·+   · *   `*`*·
# ·`··. *`• *·.        https://adventofcode.com/2025/day/9  ` ··*     `· • ·`·.*
# `·.*`··  `·. +   ·  `•`*  ··  `· .·* ·*. `·`+·     .  ·* · `   ·.· *  ·` *·.·`

import typing as t
from functools import cache

import numba
import numpy as np
from numpy.typing import NDArray

coordtype: t.TypeAlias = tuple[int, int]

_space = "."
_red_tile = "#"
_green_tile = "X"


raw = """7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3"""


def parse(s: str) -> list[coordtype]:
    res = []
    for line in s.splitlines():
        i, j = map(int, line.split(","))
        res.append((i, j))
    
    return res


@numba.njit(cache=True)
def _interpolate_numba(a: NDArray[np.int_], b: NDArray[np.int_]) -> set[coordtype]:
    """Returns the set of all coordinates on a horizontal or vertical line connecting
    a and b."""

    res = set([(i, i) for i in range(0)])

    assert a.shape == b.shape
    diff = b - a

    assert (diff != 0).sum() == 1

    diffinds = np.where(diff != 0)[0]
    assert len(diffinds) == 1

    n_steps = np.abs(diff).sum()
    delta = diff.clip(-1, +1)
    for n in range(n_steps + 1):
        point = a + n*delta
        i, j = point
        res.add((i, j))
        
    return res


@cache
def interpolate_path(at: coordtype, bt: coordtype) -> set[coordtype]:
    a, b = map(np.array, (at, bt))
    return _interpolate_numba(a, b)


def get_neighbors(x: coordtype, m: NDArray) -> t.Iterator[tuple[int, int]]:
    """Given coordinates and an array, iterates over all in-bounds neighbors of the coordinates."""
    steps = ((1, 0), (0, 1), (-1, 0), (0, -1))
    i, j = x
    for di, dj in steps:
        x = (i+di, j+dj)
        if all(0 <= c < lim for c, lim in zip(x, m.shape)):
            yield x
        #
    #


def flood_fill_enclosed(corner_coords: t.Sequence[coordtype]) -> NDArray[np.str_]:
    """Takes the coordinates of the red corner tiles.
    Returns an ascii map of the pattern resulting from laying green tiles along the circumference,
    and filling the enclosed area with green tiles."""

    # Initialize with empty space everywhere
    m = np.full(tuple(max(vals)+1 for vals in zip(*corner_coords)), _space, dtype='<U1')

    # Lay green tiles along the periphery
    for ind, a in enumerate(corner_coords):
        b = corner_coords[(ind+1) % len(corner_coords)]
        m[*zip(*interpolate_path(a, b))] = _green_tile
    
    # Put red tiles on the corners
    m[*zip(*corner_coords)] = _red_tile

    # Use flood filling to find connected regions, and add green tiles to enclosed regions 
    remaining = {(i, j) for (i, j), char in np.ndenumerate(m) if char == _space}
    while remaining:
        # Start with a random unprocessed tile
        seed = remaining.pop()
        region = set()
        front = {seed}
        # Use BFS to get the entire region connected to the seed
        while front:
            region |= front
            front = {n for site in front for n in get_neighbors(site, m) if n not in region and n in remaining}
        
        # If the region touches an edge, the region is not enclosed.
        contained = all(all(0 < coord < lim - 1 for coord, lim in zip(x, m.shape)) for x in region)
        if contained:
            m[*zip(*region)] = _green_tile
        
        remaining -= region

    return m


def make_coordinate_compression(coords: t.Sequence[coordtype]) -> dict[coordtype, coordtype]:
    """Makes a mapping from the input coordinates to a compressed representation.
    Works by considering the distinct x and y values, and enumerating both in ascending order.
    Each coordinate is then mapped to its index, so e.g. the smallest coordinate is mapped to 0, etc.
    Returns a dict mapping each original coordinate tuple to the corresponding compressed coordinate."""

    # Make a mapping from coordinates to indices
    compressed_inds = []
    coord_maps: list[dict[int, int]] = []
    for coordvals in zip(*coords):
        distinct_coords = sorted(set(map(int, coordvals)))
        compressed_inds.append(distinct_coords)
        coord_maps.append({val: i for i, val in enumerate(distinct_coords)})
    
    # Construct the mapping to compressed coordinates
    res = dict()
    for coord in coords:
        ic, jc = tuple(map_[c] for c, map_ in zip(coord, coord_maps))
        res[coord] = (ic, jc)
    
    return res


class Grid:
    def __init__(self, corners: t.Iterable[coordtype], compress=True) -> None:
        """Corners: A numpy array containing coordinates of the red tiles.
        compress indicates whether to do coordinate compression."""

        self.compress = compress
        self._corners = tuple(c for c in corners)

        # Define mapping to compressed coordinates        
        self._comp_map = make_coordinate_compression(self._corners)
        self._comp_map_rev = {v: k for k, v in self._comp_map.items()}
        assert len(self._comp_map) == len(self._comp_map_rev)
        self._corners_compressed = tuple(self._comp_map[c] for c in self._corners)

        self.corners = self._corners_compressed if self.compress else self._corners

        # Draw the tiles along the outer edge
        self.m = flood_fill_enclosed(self.corners)
        self.free = {x for x, char in np.ndenumerate(self.m) if char == _space}

    def display(self):
        lines = ["".join(row) for row in np.rot90(self.m)]
        s = "\n".join(lines)
        print(s)

    def area(self, a: coordtype, b: coordtype) -> int:
        """Computes the area of a rectangle with the two input points as its opposing corners"""
        # Recover original coordinates if using compression
        if self.compress:
            a, b = self._comp_map_rev[a], self._comp_map_rev[b]

        res = 1
        for xa, xb in zip(a, b):
            res *= (abs(xb - xa) + 1)
        
        return res
    
    def corner_pairs(self) -> t.Iterator[tuple[coordtype, coordtype]]:
        """Iterates over all pairs of corner tiles"""
        c = [(int(i), int(j)) for i, j in self.corners]
        for i, a in enumerate(c):
            for b in c[i+1:]:
                yield a, b
            #
        #

    @cache    
    def _intersect_free_space(self, a: coordtype, b: coordtype) -> bool:
        """Determines whether the line from a to b touches an un-tiled location"""
        return not self.free.isdisjoint(interpolate_path(a, b))

    def enclosed(self, a: coordtype, b: coordtype) -> bool:
        """Determines whether the rectangle with a and b as its opposing corners consists
        entirely of green tiles, i.e. whether it is enclosed by the circumference of green/red
        tiles defined by the corner tiles.
        Works by tracing the edges of the rectangle and detecting if any free space is encountered."""

        # Define the 4 corners
        ia, ja = a
        diff_i, diff_j = (xb - xa for xa, xb in zip(a, b))
        cycle = (a, (ia, ja+diff_j), b, (ia+diff_i, ja))
        
        # Walk along each edge and check for free space
        for i, point in enumerate(cycle):
            next_ = cycle[(i+1) % len(cycle)]
            if self._intersect_free_space(point, next_):
                return False
            #
        
        # If we don't intersect any free space, the entire rectangle is enclosed
        return True
    
    
def solve(data: str) -> tuple[int|str, ...]:
    red_tiles = parse(data)
    
    grid = Grid(red_tiles)
    pairs = list(grid.corner_pairs())
    pairs.sort(key=lambda tup: -grid.area(*tup))
    
    star1 = grid.area(*pairs[0])
    print(f"Solution to part 1: {star1}")

    enclosed_squares = filter(lambda p: grid.enclosed(*p), pairs)
    largest = next(enclosed_squares)

    star2 = grid.area(*largest)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 9
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
