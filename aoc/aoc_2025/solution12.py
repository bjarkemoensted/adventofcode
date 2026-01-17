# *.·. ·  ` +.  ·. ·+ *·` .·.*     .  · +.·*` ·.    ·` ·  · + *.  ·.+·  · `·.  ·
#  ·  *•·`  .  · · .·  *·`·• · Christmas Tree Farm `  +  ·  . · `·*   ·..·  *·.*
# `·  ·.`*  ·+.     *· https://adventofcode.com/2025/day/12    ·* .· *. `··. .`·
# ·+`·.`.· *.·  ·` + .·   *   ·*.`·    +` . *· · .      ·+ + ·` .        *.`·*·.

from __future__ import annotations

import functools
import typing as t
from dataclasses import dataclass
from operator import mul

import numpy as np
from numpy.typing import NDArray

_space = "."
_shape = "#"
_map = {_space: False, _shape: True}
_revmap = {v: k for k, v in _map.items()}


def array_to_int(arr: NDArray[np.bool_]) -> int:
    res = 0
    for i, val in enumerate(arr.flat):
        if val:
            res += 2**i
        #
    
    return res


def generate_transformations(m: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Takes a 2D array representing a present.
    Returns a 3D array where each slice m[i,:,:] is a distinct rotation
    obtained by flipping/rotating the present."""
    
    slices = []
    seen = set()

    for n_rot in range(4):
        m_rotated = np.rot90(m, n_rot)
        for slice in (m_rotated, np.fliplr(m_rotated)):
            key = array_to_int(slice)
            if key in seen:
                continue
            seen.add(key)
            slices.append(slice)
        #
    
    res = np.stack(slices, axis=0)

    assert res.shape == (len(slices), *m.shape)
    return res


@dataclass
class Region:
    shape: tuple[int, int]
    presents: tuple[int, ...]


def parse(s: str) -> tuple[NDArray[np.bool_], list[Region]]:
    section = s.split("\n\n")
    shapeparts = section[:-1]
    regionparts = section[-1]
    
    present_arrs = []
    regions = []

    for i, sp in enumerate(map(str.splitlines, shapeparts)):
        assert i == int(sp[0].split(":")[0])  # make sure the shapes are specified in order
        m = np.array([[_map[char] for char in line] for line in sp[1:]], dtype=bool)
        present_arrs.append(m)

    assert len({m.shape for m in present_arrs}) == 1
    
    present_shapes = (len(present_arrs), *present_arrs[0].shape)
    presents = np.full(present_shapes, dtype=bool, fill_value=False)
    for i, present in enumerate(present_arrs):
        presents[i, :, :] = present

    for rp in regionparts.splitlines():
        size_str, nums_str = rp.split(": ")
        width, height = map(int, size_str.split("x"))
        target = tuple(map(int, nums_str.split()))
        region = Region(shape=(width, height), presents=target)
        regions.append(region)

    return presents, regions


class TetrisSolver:
    """Solver for the present-layout problem.
    This contains the present shapes for the problem, and exposes helper methods for figuring out which
    additional presents it is possible to fit into a subregion.
    The main strategy is to start in the top-left corner of an entirely empty region,
    then focus on a small nxn 'active' area there (nxn is the dimension of the presents).
    Each combination of presents which fit there is computed, and stored on a queue, following which the active
    window is shifted slightly. This process continues until it becomes clear that the remaining of the desired
    combination of presents can fit, or all attempts have been exhausted."""

    def __init__(self, presents: NDArray[np.bool_], verbose=False) -> None:
        self.verbose = verbose
        self.presents = presents.copy()
        self.present_transformations = [generate_transformations(p) for p in self.presents]

        self._present_n_pixels = [p.sum() for p in self.presents]
        a, b, c = self.presents.shape
        self.n_presents = a
        self.present_shape = (b, c)
        self._fit_cache: dict[int, list[NDArray[np.bool_]]] = dict()
    
    def vprint(self, *args, **kwargs) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def fits_in_region(self, region: NDArray[np.bool_]) -> list[NDArray[np.bool_]]:
        key = array_to_int(region)
        try:
            return self._fit_cache[key]
        except KeyError:
            pass

        res_list: list[list[NDArray[np.bool_]]] = [[] for _ in self.present_transformations]
        for i, arr in enumerate(self.present_transformations):
            for trans in arr:
                fits = not np.any(region & trans)
                if fits:
                    res_list[i].append(trans.copy())
                #
            #
        
        res = [np.array(elem) for elem in res_list]
        self._fit_cache[key] = res
        return res

    def display_state(self, state: NDArray[np.bool_]) -> None:
        lines = ["".join(_revmap[char] for char in row) for row in state]
        s = "\n".join(lines)
        print(s, end="\n\n")

    def count_pixels(self, present_counts: NDArray[np.int_]) -> int:
        """Counts the number of 'pixels' (1x1 sites occupied by a gift) for the input combination
        of gifts. The input is an array of counts, so [1, 3, 2] means 1 of the first present type,
        3 of the second, and so on."""
        res = 0
        assert len(present_counts) == len(self._present_n_pixels)
        for i, n in enumerate(present_counts):
            res += n*self._present_n_pixels[i]
        
        return int(res)

    def _iter_active_regions(self, target_shape: tuple[int, int]) -> t.Iterator[tuple[int, int]]:
        """Iterates over the top-left corners of all active regions for the specified target shape"""
        
        present_height, present_width = self.present_shape
        rows, cols = target_shape
        i, j = 0, 0
        while True:
            yield i, j
            j += 1
            if j + present_width > cols:
                j = 0
                i += 1
            if i + present_height > rows:
                return
            #
        #

    def brute_force(self, target_shape: tuple[int, int], presents: NDArray[np.int_]) -> bool:
        """Use a brute force approach to check if the specified presents fit in the target shape."""
        
        if presents.sum() == 0:
            return True
        # Start the active region in the top-left corner
        map_initial = np.full(shape=target_shape, fill_value=False, dtype=bool)
        
        # map, remaining
        initial_state = (map_initial, presents.copy())
        queue = [initial_state]

        present_height, present_width = self.present_shape
        rows, cols = target_shape
        nits = 0

        for i, j in self._iter_active_regions(target_shape):
            nits += 1
            self.vprint(f"Iteration: {nits}, region: {(i, j)}")
            next_ = []

            for state in queue:
                map_, remaining = state
                if remaining.sum() == 0:
                    return True
                
                pixels_left = np.sum(~map_[i:i+present_height, j:]) + max(0, rows-i-present_height)*cols
                pixels_needed = self.count_pixels(remaining)
                hopeless = pixels_left < pixels_needed
                if hopeless:
                    continue
                active_region = map_[i:i+present_height, j:j+present_width]
                next_.append((map_, remaining))
                fits_here = self.fits_in_region(active_region)
                missing, *_ = np.nonzero(remaining)
                for ind in missing:
                    new_remaining = remaining.copy()
                    new_remaining[ind] -= 1
                    done = new_remaining.sum() == 0

                    for part in fits_here[ind]:
                        if done:
                            return True
                        new_map = map_.copy()
                        new_map[i: i+present_height, j: j+present_width] |= part
                        
                        new_state = (new_map, new_remaining)
                        next_.append(new_state)
                    #
                #

            queue = next_
        
        return False


    def check_simple_cases(self, target_shape: tuple[int, int], presents: NDArray[np.int_]) -> bool|None:
        """Checks for simple cases where a) the presents packed with zero space exeec the available area, or b)
        the target region is so large the presents can fit without any overlap (each nxn present shape can get
        its own nxn grid)"""
        
        # If the non-empty pixels in the presents exceed the available area, a fit is impossible
        area = functools.reduce(mul, target_shape)
        area_required_with_perfect_packing = self.count_pixels(present_counts=presents)
        if area_required_with_perfect_packing > area:
            return False
        
        # If the presents fit in independent present-sized grids, a fit is easily possible
        n_presents = sum(presents)
        n_pres_by_len = (len_ // pres for len_, pres in zip(target_shape, self.present_shape, strict=True))
        n_independent_squares = functools.reduce(mul, n_pres_by_len)
        if n_independent_squares >= n_presents:
            return True
        
        # Otherwise, it's tricky
        return None

    def will_it_fit(self, target_shape: tuple[int, int], presents: t.Sequence[int]) -> bool:
        """Determines whether the input target shape can contain the input combination of presents."""

        presents_arr = np.array(presents)
        simple = self.check_simple_cases(target_shape=target_shape, presents=presents_arr)
        if isinstance(simple, bool):
            return simple
        else:
            return self.brute_force(target_shape=target_shape, presents=presents_arr)


def solve(data: str) -> tuple[int|str|None, ...]:
    shapes, regions = parse(data)
    ts = TetrisSolver(shapes, verbose=False)

    star1 = sum(ts.will_it_fit(region.shape, region.presents) for region in regions)
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 12
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
