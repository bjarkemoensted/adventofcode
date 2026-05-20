# `·*.`   . ·*·  ` .·  *`  ·   * ·  ` .* `·.    *·    ·    ·*.` ·   +·.  ·.`*·.·
#   `·.·`  *+·* `·.   *.    ·`  Planet of Discord · .*`     ·*.`  ·    · ` · .·*
# ·`. ·*.`·     ·      https://adventofcode.com/2019/day/24   ··  *.`     ·*·`.`
# +··`*  .   `   . ·`*·`+*· .·  `*  . ·  *   ·  .·*. · `* ·     *` ·    · *.`  ·

from collections import defaultdict
from functools import cache
from typing import Iterator, Self

import numpy as np
from numpy.typing import NDArray

# Location type: the level in the recursive structure, plus i, j coords
type loc = tuple[int, tuple[int, int]]
bug_lookup = {"#": True, ".": False}


def parse(s: str) -> NDArray[np.bool_]:
    res = np.array([[bug_lookup[char] for char in line] for line in s.splitlines()])
    return res


class Simulation:
    """Represents a simulation of Eris-bugs.
    Because part 2 concerns an infinite recursive space, states are represented
    by a set of currently alive bugs. The number of nonzero bug neighbors
    can then be found be considering only the adjacent squares to each bug,
    as the remainder of the space must be empty."""

    def __init__(self, arr: NDArray[np.bool_], recursive=False) -> None:
        self.recursive = recursive
        self.shape = arr.shape
        assert all(n % 2 == 1 for n in self.shape)
        self.rows, self.cols = self.shape
        self.midpoint = (self.rows // 2, self.cols // 2)
        
        self.state: set[loc] = set()
        self.seen: set[frozenset[tuple[int, tuple[int, int]]]] = set()
        self.initialize_state(arr=arr)
    
    def key(self):
        """Hashable key for current state, to check for recurrences"""
        res = frozenset(self.state)
        return res

    def initialize_state(self, arr: NDArray[np.bool_]) -> None:
        self.state.clear()
        self.seen.clear()
        for (i, j), bug in np.ndenumerate(arr):
            if bug:
                self.state.add((0, (i, j)))
            #
        #

    def _recurse_in(self, level: int, i_prev: int, j_prev: int) -> Iterator[loc]:
        """Handles the case where wi hit the middle square. If not recursive, we simply get the middle square.
        If recursive, we get the entire right side if we stepped left into the middle square, etc"""

        if not self.recursive:
            yield level, self.midpoint
            return
        
        im, jm = self.midpoint
        i_range = range(self.rows)
        j_range = range(self.cols)
        if i_prev != im:
            row = 0 if i_prev < im else self.rows - 1
            yield from ((level+1, (row, c)) for c in j_range)
        elif j_prev != jm:
            col = 0 if j_prev < jm else self.cols - 1
            yield from ((level+1, (c, col)) for c in i_range)
        else:
            raise RuntimeError

    def _recurse_out(self, level: int, i: int, j: int) -> Iterator[loc]:
        """Case when we look outside the map. If recursive, generates the target point one level up"""
        if not self.recursive:
            return
        
        p = (i, j)
        target = list(self.midpoint)
        for ind, c in enumerate(p):
            if c < 0:
                target[ind] -= 1
            elif c >= self.shape[ind]:
                target[ind] += 1
            #
        ti, tj = target
        assert (ti, tj) != self.midpoint
        yield level-1, (ti, tj)
    
    def iter_neighbors(self, pos: loc) -> Iterator[loc]:
        """Iterate over the neighbors to the input level and ij coords"""
        level, (i, j) = pos
        for p in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            ip, jp = p
            out_of_bounds = not all(0 <= x < lim for x, lim in zip(p, self.shape))

            if p == self.midpoint:
                # handle cases where we hit the middle square and might have to recurse
                yield from self._recurse_in(level, i, j)
            elif out_of_bounds:
                # handle cases were we go out of bounds and possible recurse out
                yield from self._recurse_out(level, ip, jp)
            else:
                yield level, p
            #
        #

    @cache
    def get_neighbors(self, pos: loc) -> tuple[loc, ...]:
        res = tuple(self.iter_neighbors(pos))
        return res

    def count_adjacent(self) -> dict[loc, int]:
        """Count the number of adjacent bugs for all cells. Returns a dict with the non-zero counts"""
        res: dict[loc, int] = defaultdict(int)
        for pos in self.state:
            for adj in self.get_neighbors(pos):
                res[adj] += 1
            #
        
        return res

    def _ascii_array(self, at_level=0) -> NDArray[np.str_]:
        rev = {v: k for k, v in bug_lookup.items()}
        arr = np.full(shape=self.shape, fill_value=rev[False])
        for level, (i, j) in self.state:
            if level == at_level:
                arr[i, j] = rev[True]
            #
        if self.recursive:
            arr[*self.midpoint] = "?"
        return arr

    def display(self, at_level: int | None = None) -> None:
        _distinct = {level for level, _ in self.state}
        levels = list(range(min(_distinct), max(_distinct)+1)) if at_level is None else [at_level]
        
        for level in levels:
            if len(levels) > 1:
                print(f"Depth {level}:")
            arr = self._ascii_array(at_level=level)
            s = "\n".join(["".join(row) for row in arr])
            print(s, end="\n\n")
        #

    def _tick(self):
        """Simulate a single iteration"""
    
        # Sum up the number of adjacent bugs around each existing bug
        bug_counts = self.count_adjacent()
        
        # Determine which bugs are dying and which new sites are being infested
        deaths = {pos for pos in self.state if bug_counts[pos] != 1}
        infestations = {pos for pos, n in bug_counts.items() if pos not in self.state and n in (1, 2)}
        
        # Update current state accordingly
        self.state -= deaths
        self.state |= infestations
        
    def run(self, n=-1) -> Self:
        """Simulate n iterations, or, if n == -1, until a recurrence is detected"""
        nits = 0
        done = False
        self.seen.add(self.key())
        
        while not done:
            self._tick()
            key = self.key()
            nits += 1
            if n == -1:
                done = key in self.seen
            else:
                done = nits >= n
            self.seen.add(key)
        return self

    def biodiversity_rating(self, at_level=0) -> int:
        """Computes the biodiversity rating at the specified level"""
        res = 0
        factor = 1
        for i, j in np.ndindex(self.shape):
            
            pos = (at_level, (i, j))
            if pos in self.state:
                res += factor
            factor *= 2
        
        return res

    def count_bugs(self) -> int:
        return len(self.state)


def solve(data: str) -> tuple[int|str, ...]:
    arr = parse(data)

    star1 = Simulation(arr=arr).run().biodiversity_rating()
    print(f"Solution to part 1: {star1}")

    star2 = Simulation(arr=arr, recursive=True).run(n=200).count_bugs()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 24
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
