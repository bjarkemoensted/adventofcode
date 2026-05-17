# .·`* `  ·* ·     +·  ·. `.  ·`•  .·` *·`      ·· *  ·    .*`  ·*· ·   *.· `.`·
# ·+· ` ·. ·   •. · `*  ··    `  · Tractor Beam   `.  *· ·    · `·*.   ` · · ·.*
# * .`·*.·  `* .··   · https://adventofcode.com/2019/day/19     *  `.·*   `· *·`
# ··• *·`+ `*  ·+  · .  ` * ·  ·. ` .  + ·· ` . • · *·.`  ·   . ·`  + · ·`  ··*.

import math
from dataclasses import dataclass
from typing import Callable

from aoc.aoc_2019.intcode import Computer

type coord = tuple[int, int]
# Assuming since part 1 prompts for the initial 50x50 grid that no slope greater than that occurs
UPPER_BOUND = 50


def parse(s: str) -> list[int]:
    return list(map(int, s.split(",")))


@dataclass
class Slope:
    """Helper class for representing a line slope and updating it with
    additional information. It stores bounds on the slope a of an edge of the
    tractor beam:
    j = a*i
    """

    lower: float=0.0
    upper: float=float("inf")
    fitted: bool=False
    _i_max: int = -1

    def add_observation(self, i: int, j: int) -> None:
        """Adds an observed point i, j"""
        eps = 0.5
        if i == 0 or i < self._i_max:
            return
        
        self.lower = (j - eps) / (i + eps)
        self.upper = (j + eps) / (i - eps)
        self.fitted = True
        
        assert self.upper >= self.lower
        
    def predict(self, i: int) -> tuple[float, float]:
        """Predict the value of j at the input i coordinate.
        The result represents the lower and upper bounds on the value of j"""
        assert self.fitted
        lower = i*self.lower
        upper = i*self.upper

        return lower, upper
    #

def bisection_search(x0: int, func: Callable[[int], bool], left=True) -> int:
        """Uses bijection search to locate the two subsequent integers a, b where
        the input callable evaluates to (False, True) (if left) or (True, False) (if right).
        func: The callable to use.
        x0: a point where we assume (func(x0) = True)"""

        assert func(x0)

        delta = -1 if left else +1
        bounds = [x0, x0]
        ind = 0 if left else 1
        other_ind = 1 - ind
        bounds[ind] = x0 + delta

        # Move other point until it's definitely outside the beam
        while func(bounds[ind]):
            if bounds[ind] < 0:
                return 0
            delta *= 2
            bounds[ind] = x0 + delta
                
        # Cut search space in half until (left, right) captures the beam edge
        while bounds[1] - bounds[0] > 1:
            mid = (bounds[1] + bounds[0]) // 2
            update_ind = int(not (func(mid) ^ left))
            bounds[update_ind] = mid
        
        # Double check the results
        assert func(bounds[other_ind])
        assert not func(bounds[ind])

        res = bounds[other_ind]

        return res


class Scanner:
    """A scanner helper class for determining tractor beam properties"""

    def __init__(self, program: list[int]) -> None:
        self._program = [val for val in program]
        # cache points inside/outside beam
        self._cache: dict[coord, bool] = dict()
        # cache beam left/right edge at each i
        self._edges_cache: dict[int, tuple[int, int]] = dict()
        
        # Lower and upper bounds on the left and right slopes of the beam
        self.edge_slopes = [Slope(), Slope()]

        # Start by finding a point on the rim of a square around origin
        cut = UPPER_BOUND - 1
        _initial_rim = sum(([(cut, cut-a), (cut-a, cut)] for a in range(cut)), [])[1:]
        y0, x0 = next(p for p in _initial_rim if self.inside_beam(*p))
        # Determine the beam edges from that point, so we have some info on the beam edge slopes
        self._probe_line(y0, x0)

    def _margin(self, i: int, left=True) -> int|None:
        """Use known bounds on the slopes of the beam's edges to find
        a point inside the tractor beam on the input line.
        Returns None if no pixels are affected by the beam (can happen close
        to the origin due to rounding)."""

        a, b = self.edge_slopes[0 if left else 1].predict(i)
        eps = 1

        # Determine bounding points to consider
        if left:
            points = [val for val in range(math.ceil(a), math.ceil(b)+eps) if val >= 0]
        else:
            points = [val for val in range(math.floor(a), math.ceil(b)+eps) if val >= 0]
        if not left:
            points.reverse()
        
        # Determine one of the points, return None if all points are outside the beam
        try:
            res = next(p for p in points if self.inside_beam(i, p))
        except StopIteration:
            return None

        return res

    def inside_beam(self, i: int, j: int) -> bool:
        """Returns whether the input point is inside the tractor beam"""
        
        if i < 0 or j < 0:
            raise ValueError("Coordinates must be non-negative")
        try:
            return self._cache[(i, j)]
        except KeyError:
            pass

        status = Computer(self._program).add_input(j, i).run().read_stdout()
        assert status in (0, 1)
        res = status == 1
        self._cache[(i, j)] = res
        return res

    def get_beam_edges(self, i: int) -> tuple[int, int]:
        """Determine the j-coordinate of the leftmost and rightmost point inside
        the tractor beam at line i."""
        
        # Used cached result if available
        if i in self._edges_cache:
            return self._edges_cache[i]
        
        # Use known slope bounds to determine a search region
        left_bound = self._margin(i, left=True)
        right_bound = self._margin(i, left=False)
        if left_bound is None:
            assert right_bound is None
            raise RuntimeError(f"No beam at line {i}")

        # Use bisection search to find the edges
        res = self._probe_line(i=i, left_bound=left_bound, right_bound=right_bound)
        self._edges_cache[i] = res
        return res

    def _probe_line(self, i: int, left_bound: int, right_bound: int|None=None) -> tuple[int, int]:
        """Scans across the input line (specified by i-coordinate) to find the left and right
        edges of the tractor beam.
        left_bound: A point just to the right of the beam's left edge.
        right_bound: Optional - a point just to the left of the right edge.
            If not provided, the same point will be used"""
        
        right_bound = left_bound if right_bound is None else right_bound
        
        # Edges of the beam at this i-value
        these_edges = [-1, -1]

        for left in (True, False):
            slope_ind = 0 if left else 1
            bound = left_bound if left else right_bound
            assert self.inside_beam(i, bound)
            edge = bisection_search(x0=bound, func=lambda x: self.inside_beam(i=i, j=max(0, x)), left=left)
            self.edge_slopes[slope_ind].add_observation(i=i, j=edge)
            
            these_edges[slope_ind] = edge
        
        a, b = these_edges
        self._edges_cache[i] = (a, b)
        return a, b

    def count_affected_in_region(self, width: int) -> int:
        """Counts the number of coordinates in the initial square region with
        y and x < width which are affected by the tractor beam"""
        
        res = 0
        for i in range(width):
            # Catch the runtime error for the lines with no beam pixels
            try:
                edges = self.get_beam_edges(i)
            except RuntimeError:
                continue
            
            # Add the number of affected pixels inside the square on this line
            a, b = edges
            if a >= width:
                continue
            a, b = (min(width-1, val) for val in edges)
            res += (b - a + 1)

        return res

    def _square_overhead(self, i: int, width: int) -> int:
        """If inserting a square with the specified width at line i in the tractor beam,
        return the amount of overhead room left.
        If there's obviously no room for the square, -width is returned."""
        
        if i < width - 1:
            return -width
        
        a, b = self.get_beam_edges(i)

        right = a + width - 1
        # the lower beam slice must accomodate the entire region
        if b < right:
            return -width

        _, b2 = self.get_beam_edges(i - width + 1)
        delta = b2 - right
        return delta

    def find_square_location(self, width: int) -> coord:
        """Determines the northwestern corner of the first coordinate where a square
        of the input width can fit entirely inside the tractor beam.
        Because some rounding logic makes a bit of noise on the edges of the beam, we first
        use bisection search to find the first line where the square can 'almost' fit,
        meaning the beam width is 2 too narrow (because rounding can make the edges differ by
        1 each). Then a more granular scan is made until a location with enough space is found."""
        
        # look for the first line with an overhead of -2
        def func(x: int) -> bool:
            return self._square_overhead(x, width=width) >= -2
        
        # Keep incrementing x0 until we overshoot
        x0 = 2**math.ceil(math.log2(width-1))
        while not func(x0):
            x0 *= 2
        
        # Bisection search to find a line where the square almost fits
        i_south = bisection_search(x0=x0, func=func, left=True)
        
        # More granular scan for the exact location
        while self._square_overhead(i_south, width=width) < 0:
            i_south += 1

        # Determine the coordinates for the upper left corner of the square
        i_corner = i_south - width + 1
        _, b = self.get_beam_edges(i_corner)
        j_corner = b - width + 1

        return i_corner, j_corner
    

def solve(data: str) -> tuple[int|str, ...]:
    program = parse(data)

    scanner = Scanner(program)

    star1 = scanner.count_affected_in_region(width=50)
    print(f"Solution to part 1: {star1}")

    y_square, x_square = scanner.find_square_location(100)    
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
