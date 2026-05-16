# .·`* `  ·* ·     +·  ·. `.  ·`•  .·` *·`      ·· *  ·    .*`  ·*· ·   *.· `.`·
# ·+· ` ·. ·   •. · `*  ··    `  · Tractor Beam   `.  *· ·    · `·*.   ` · · ·.*
# * .`·*.·  `* .··   · https://adventofcode.com/2019/day/19     *  `.·*   `· *·`
# ··• *·`+ `*  ·+  · .  ` * ·  ·. ` .  + ·· ` . • · *·.`  ·   . ·`  + · ·`  ··*.

from collections import defaultdict, deque
from itertools import product
from typing import Iterator

import numpy as np

from aoc.aoc_2019.intcode import Computer

type coord = tuple[int, int]
# Assuming since part 1 prompts for the initial 50x50 grid that no slope greater than that occurs
UPPER_BOUND = 50

def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


def _iter_square(startat=0, stopat=-1) -> Iterator[coord]:
    """Iterate over the points at the edge of a squate of ever-increasing side length"""
    
    assert stopat == -1 or stopat > startat
    n = startat

    while stopat == -1 or n < stopat:
        yield n, n
        for other in range(n-1, -1, -1):
            yield n, other
            yield other, n
            
        n += 1


def _determine_slope_bounds(i: int, j: int, i0=0, j0=0) -> tuple[float, float]:
    """Determine the bounds on a slope (dx/dy because we're using matrix coords),
    given points i, j (and optionally i0, j0, defaulting to 0) on the line.
    Works by assuming a true, continuous slope like
    x_float = alpha_float*y,
    and computing bounds where x_float would be rounded to the input values."""

    di = i - i0
    dj = j - j0
    if di == 0:
        return (float("nan"), float("nan"))
    
    eps = 0.49
    lower = (dj - eps) / di
    upper = (dj + eps) / di
    return (lower, upper)


class Scanner:
    def __init__(self, program: list[int]) -> None:
        # cache points inside/outside beam
        self._cache: dict[coord, bool] = dict()
        
        # cache beam left/right edge at each i
        self._edges_cache: dict[int, tuple[int, int]] = dict()
        self.max_y_probed = -1
        
        self.program = [val for val in program]
        # Lower and upper bounds on the left and right slopes of the beam
        self.edge_slopes = [[float("nan"), float("nan")], [float("nan"), float("nan")]]

        

        assert self.inside_beam(0, 0)

        cut = 20
        square = _iter_square(stopat=cut)
        self.m = np.zeros(shape=(cut, cut), dtype=int)
        for x, y in square:
            if self.inside_beam(x, y):
                self.m[x, y] = 1
            #

        temp = defaultdict(list)
        for (i, j), inside in self._cache.items():
            if not inside:
                continue
            temp[i].append(j)

        for i, vals in temp.items():
            mid = sum(vals) // len(vals)
            self.probe_line(i, mid)
            
        self._check(cut=cut)
    
    def _check(self, cut: int) -> None:
        # checks  TODO REMOVE !!!
        for i in range(cut):
            row = [j_ for (i_, j_ ), inside in self._cache.items() if inside and i_ == i]
            if not row:
                continue
            j0 = (min(row) + max(row)) // 2
            #print(f"checking {i=}, {j0}")

            left = self._find_edge(i, j0, left_edge=True)
            assert self.inside_beam(i, left)
            assert left == 0 or not self.inside_beam(i, left-1)

            right = self._find_edge(i, j0, left_edge=False)
            assert self.inside_beam(i, right)
            assert not self.inside_beam(i, right+1)

    def inside_beam(self, i: int, j: int) -> bool:
        """Returns whether the input point is inside the tractor beam"""
        
        if i < 0 or j < 0:
            raise ValueError("Coordinates must be non-negative")
        try:
            return self._cache[(i, j)]
        except KeyError:
            pass

        computer = Computer(self.program)
        status = computer.add_input(j, i).run().read_stdout()
        assert status in (0, 1)
        res = status == 1
        self._cache[(i, j)] = res
        return res

    def probe_line(self, i: int, left_bound: int, right_bound: int|None=None) -> None:
        """Scans across the input line (specified by i-coordinate) to find the left and right
        edges of the tractor beam.
        left_bound: A point just to the right of the beam's left edge.
        right_bound: Optional - a point just to the left of the right edge.
            If not provided, the same point will be used"""
        
        right_bound = left_bound if right_bound is None else right_bound
        most_precise = i > self.max_y_probed
        if most_precise:
            self.max_y_probed = i

        # Edges of the beam at this i-value
        these_edges = [-1, -1]

        for left in (True, False):
            slope_ind = 0 if left else 1
            bound = left_bound if left else right_bound
            assert self.inside_beam(i, bound)
            edge = self._find_edge(i, j0=bound, left_edge=left)
            these_edges[slope_ind] = edge
            bounds = _determine_slope_bounds(i=i, j=edge)
            if most_precise:
                self.edge_slopes[slope_ind][:] = bounds
            #
        
        a, b = these_edges
        self._edges_cache[i] = (a, b)

    def _find_edge(self, i: int, j0: int, left_edge=True) -> int:
        """Uses bijection search to locate the left edge of the beam.
        right: a point assumed to be inside the beam somewhere.
        left_edge: Whether to search for the left edge of the beam. Set to False to fint right edge."""

        assert self.inside_beam(i, j0)
        # Move leftmost point left until it's definitely outside the beam
        
        # If finding the left edge, the search region must go from outside-inside beam and vise versa
        invariant = [False, True] if left_edge else [True, False]

        delta = -1 if left_edge else +1
        bounds = [j0, j0]
        ind = 0 if left_edge else 1
        other_ind = 1 - ind
        bounds[ind] = j0 + delta

        # TODO NUKE THIS!!!
        def _check_invariant() -> bool:
            res = all(constraint == self.inside_beam(i, b) for constraint, b in zip(invariant, bounds, strict=True))
            return res

        while self.inside_beam(i, max(0, bounds[ind])):
            if bounds[ind] < 0:
                return 0
            assert not _check_invariant()
            delta *= 2
            bounds[ind] = j0 + delta

        assert _check_invariant()
                
        # Cut search space in half until (left, right) captures the beam edge
        while bounds[1] - bounds[0] > 1:
            mid = (bounds[1] + bounds[0]) // 2
            midpoint_in_beam = self.inside_beam(i, mid)
            update_ind = int(not (midpoint_in_beam ^ left_edge))
            bounds[update_ind] = mid

            if not _check_invariant():
                print(f"{midpoint_in_beam=}, {left_edge=}, new bounds: {bounds}")
                raise RuntimeError
            #
        
        assert self.inside_beam(i, bounds[other_ind])
        assert not self.inside_beam(i, bounds[ind])
        return bounds[other_ind]

    def expand(self) -> None:
        """Detemine the beam edges at the next line"""
        new_edges = [-1, -1]
        i_current = self.max_y_probed
        i = i_current + 1

        for left_edge in (True, False):
            ind = 0 if left_edge else 1
            lower, upper = self.edge_slopes[ind]

            # extrapolate using the known slope bounds
            est_low = round(i*lower)
            est_high = round(i*upper)
            
            predictable = est_low == est_high
            j0 = est_high if left_edge else est_low
            if predictable:
                # TODO FIX
                cut = self._find_edge(i=i, j0=est_low, left_edge=left_edge)
                print(cut, est_low)
            else:
                cut = self._find_edge(i=i, j0=j0, left_edge=left_edge)
            
            new_bounds = _determine_slope_bounds(i, cut)

            self.edge_slopes[ind][:] = new_bounds
            new_edges[ind] = cut
            
        
        self.max_y_probed = i
        a, b = new_edges
        self._edges_cache[i] = (a, b)

    def find_room_for_square(self, width=100) -> coord:
        widths: deque[tuple[int, int]] = deque(maxlen=width)

        def done() -> tuple[int, int]|None:
            nonlocal widths
            if len(widths) < width:
                return None
            a1, right_edge = widths[0]
            left_edge = right_edge - width + 1
            if left_edge < a1:
                return None
            
            a2, _ = widths[-1]
            if a2 <= left_edge:
                return left_edge, right_edge
            else:
                return None
            #

        i = -1
        while True:
            i += 1
            while i > self.max_y_probed:
                self.expand()
            
            if i not in self._edges_cache:
                widths.clear()
                continue
            
            slice_width = self._edges_cache[i]
            widths.append(slice_width)

            side_locations = done()
            if side_locations is not None:
                a, b = side_locations
                assert all(le <= a and re >= b for le, re in widths)
                
                return i - width + 1, a
            #
        #
    #


def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)

    scanner = Scanner(program)

    points = ((i, j) for i, j in product(range(50), repeat=2))

    star1 = sum(scanner.inside_beam(*p) for p in points)
    print(f"Solution to part 1: {star1}")

    y_square, x_square = scanner.find_room_for_square(100)
    star2 = 10_000*x_square + y_square
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 19
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
