# *.·. ·  ` +.  ·. ·+ *·` .·.*     .  · +.·*` ·.    ·` ·  · + *.  ·.+·  · `·.  ·
#  ·  *•·`  .  · · .·  *·`·• · Christmas Tree Farm `  +  ·  . · `·*   ·..·  *·.*
# `·  ·.`*  ·+.     *· https://adventofcode.com/2025/day/12    ·* .· *. `··. .`·
# ·+`·.`.· *.·  ·` + .·   *   ·*.`·    +` . *· · .      ·+ + ·` .        *.`·*·.

from __future__ import annotations
from dataclasses import dataclass
import functools
import numba
import numpy as np
from numpy.typing import NDArray
from operator import mul
import typing as t

shapetype: t.TypeAlias = NDArray[np.int_]


_space = "."
_shape = "#"
_map = {_space: 0, _shape: 1}
_revmap = {v: k for k, v in _map.items()}

raw = """0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2"""


@dataclass
class Region:
    shape: tuple[int, int]
    presents: tuple[int, ...]


def parse(s: str) -> tuple[list[shapetype], list[Region]]:
    section = s.split("\n\n")
    shapeparts = section[:-1]
    regionparts = section[-1]
    
    shapes = []
    regions = []

    for i, sp in enumerate(map(str.splitlines, shapeparts)):
        assert i == int(sp[0].split(":")[0])  # make sure the shapes are specified in order
        m = np.array([[_map[char] for char in line] for line in sp[1:]], dtype=int)
        shapes.append(m)

    for rp in regionparts.splitlines():
        size_str, nums_str = rp.split(": ")
        width, height = map(int, size_str.split("x"))
        target = tuple(map(int, nums_str.split()))
        region = Region(shape=(width, height), presents=target)
        regions.append(region)

    return shapes, regions


# !!!
#@numba.njit(cache=True)
def _binary_arr_as_int(m: NDArray[np.int_]) -> int:
    """Represents a binary array as a single int, by considering the
    array's elements as the binary representation of the integer"""

    x = 0
    for b in m.ravel():
        x = (x << 1) | b

    return x


@functools.cache
def _int_as_binary_arr(n: int, shape: tuple[int, int]) -> NDArray[np.int_]:
    dim = functools.reduce(mul, shape, 1)
    bits = np.array([(n >> d) & 1 for d in range(dim)])
    res = np.reshape(bits, shape)

    return res


def _binary_arr_as_str(m: NDArray[np.int_]) -> str:
    lines = ["".join([_revmap[char] for char in line]) for line in m]
    res = "\n".join(lines)
    return res


def generate_signatures(m: NDArray) -> list[int]:
    res = []
    for n_rot in range(4):
        m_rotated = np.rot90(m, n_rot)
        rs = _binary_arr_as_int(m_rotated)
        res.append(rs)
        fs = _binary_arr_as_int(np.fliplr(m_rotated))
        res.append(fs)
    
    res = list(set(map(int, res)))
    return res


class TetrisSolver:
    """Helper class for working out whether some combination of present shapes fit in a given area.
    To enable efficient caching, we represent each possible present shape by a 'signature' -  a distinct integer,
    by considering the present as a numpy array with values of 1/0 corresponding to #/.,
    then considering each element a digit in the binary representation of the number.
    For a 3x3 pixel present, each element correspond to the following powers of 2:
    0 1 2
    3 4 5
    6 7 8
    Hence, every possible shape is represented by an integer in the range 0-512.
    This also allows for overlap detection that is both efficient and allows caching, as 2 such
    arrays will have overlapping 'on' (#) pixels if and only if the bitwise 'and' operation
    on the two signature integers are nonzero.
    We can then (hopefully) brute-force solutions for smaller regions (starting with 3x3), then determine
    for each solution (combination of presents that fit) the solution's "contour" - the signature of every
    3x3 area along its east+south border. For each such contour, we can discard solutions which are Pareto-dominated
    by other solutions, meaning we drop a solution A, if another solution B fits at least as many of
    every present shape."""

    def __init__(self, shapes: t.Iterable[shapetype]) -> None:
        """Sets up a TetrisSolver instance to handle the input present shapes."""

        self.presents = tuple(shape.copy() for shape in shapes)
        self.n_presents = len(self.presents)
        # How many 'pixels' each present takes up
        self.present_sizes = [int(shape.sum()) for shape in self.presents]
        self.present_shape = self.presents[0].shape
        assert all(m.shape == self.present_shape for m in self.presents)
        self.n_bits = functools.reduce(mul, self.present_shape, 1)

        # For each present, store all its possible signatures (from flips/rotations)
        self.present_sigs = [generate_signatures(p) for p in self.presents]
        # Maps the signature of each rotated/flipped present to the index of the present
        self.shape_ints: dict[int, int] = dict()
        for i, s in enumerate(shapes):
            for sig in generate_signatures(s):
                assert sig not in self.shape_ints
                self.shape_ints[sig] = i
            #
        
        # Store at index i the signatures for present i
        self.symmetries: list[list[int]] = [[] for _ in self.presents]
        for k, v in self.shape_ints.items():

            self.symmetries[int(v)].append(int(k))
        
        # Map each signature to a 'counter' - vector with a '1' at the index of the corresponding shape
        self.counts = {k: np.array([int(i == v) for i in range(self.n_presents)]) for k, v in self.shape_ints.items()}
        
        # For each possible signature, store the signatures of presents which fit in the corresponding array
        _max_sig = 2**self.n_bits
        self._fitting_signatures: list[list[int]] = [[] for _ in range(_max_sig)]
        self.arrays = []

        for n in range(_max_sig):
            self.arrays.append(_int_as_binary_arr(n, shape=self.present_shape))
            fits = sorted({sig for sig, _ in self.shape_ints.items() if not (sig & n)})
            if fits:
                self._fitting_signatures[n] = fits
            #
        
        # For each grid size, map every contour to the Pareto frontier of possible present signatures
        self.cache: dict[tuple[int, int], dict[tuple[int, ...], NDArray[np.int_]]] = dict()
    
    def _iter_brute(self, seed_signature: int, running: NDArray[np.int_]) -> t.Iterator[tuple[int, NDArray[np.int_]]]:
        """Takes a 'seed' signature, representing the grid as it looks before any additional presents have been placed.
        Also takes a running counter, representing the number of each present type added.
        Iterates over over all possible new signatures and counters, including the situation where nothing is added."""

        # Handle the 'boring' case where we do nothing
        yield seed_signature, running

        # Recurse on each possible added present
        fits = self._fitting_signatures[seed_signature]
        for other_sig in fits:
            assert (other_sig & seed_signature) == 0  # !!!
            new_seed = other_sig + seed_signature
            inc = self.counts[other_sig]
            new_counts = running + inc
            yield from self._iter_brute(new_seed, new_counts)
    
    @functools.cache
    def brute_force_single(self, seed_signature: int) -> list[tuple[int, NDArray[np.int_]]]:
        """Given a signature, returns a list of all possible signatures and correpsonding counters (increments to
        each present type) possible from the starting situation described by that signature."""

        running = np.array([0 for _ in self.presents], dtype=int)
        res = []
        used = set()  # Store hashable representation of each solution, to avoid double counting
        for seed, arr in self._iter_brute(seed_signature=seed_signature, running=running):
            key = (seed, tuple(map(int, arr)))
            if key in used:
                continue
            used.add(key)
            res.append((seed, arr))
        
        return res
    
    @functools.cache
    def pan(self, array_signature: int, n_right: int=0, n_down: int=0) -> int:
        """considers the input array signature, and returns the signature of the array resulting
        from 'panning' the specified steps right and down.
        For example, if the input signatute corresponds to
        1 0 1
        0 1 1
        1 1 0
        and we pan one step to the right, we get
        0 1 0
        1 1 0
        1 0 0"""

        arr = self.arrays[array_signature].copy()
        if n_right:
            arr = np.roll(arr, shift=-n_right, axis=1)
            arr[:, -n_right:] = 0
        if n_down:
            arr = np.roll(arr, shift=-n_down, axis=0)
            arr[-n_down:, :] = 0
        
        res = _binary_arr_as_int(arr)
        return res

    def fits(self, shape: tuple[int, int], presents: t.Sequence[int]) -> bool:
        if any(v == 0 for v in shape):
            return sum(presents) == 0
        area = functools.reduce(mul, shape)
        pixels_on = sum(n*self.present_sizes[i] for i, n in enumerate(presents))
        if area < pixels_on:
            return False
        
        # TODO we need to start with 3x3 (present shape) grid, brute force, then expand until we reach the required shape !!!
        raise NotImplementedError

def solve(data: str) -> tuple[int|str, ...]:
    shapes, regions = parse(data)

    tetris = TetrisSolver(shapes=shapes)
    
    tetris.brute_force_single(seed_signature=0)

    star1 = -1
    print(f"Solution to part 1: {star1}")

    star2 = -1
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 12
    from aocd import get_data
    #raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
