# `· ·* +. ··      · *   .  ·*·   `·  * · ` ·+ .` *··    *• · `*·.·  * ·.·   ·+ 
# *·  ``·  .+ ·•  *`  · * ·. ·+  `  Snailfish ` +·   ·. ·    ·*  *.·`     * ·.·`
# ·.  · `·.*  *  `· ·. https://adventofcode.com/2021/day/18 *.·· ` . ·•·*`.·* `·
# .`· *+  ··   . ·`    +·· +.`  ·.  ·* ·`  .   + .· `*·     · .`*  •·  *· `  .··

import json
import math
from copy import deepcopy
from functools import reduce
from operator import mul
from typing import Iterator, cast

type nested = list[int|nested]
type nested_tuple = tuple[int|nested_tuple, int|nested_tuple]

EXPLODE_DEPTH = 4
SPLIT_THRESHOLD = 10


def parse(s):
    res = [json.loads(line) for line in s.splitlines()]
    return res


def _access_nested(arr, inds):
    ptr = arr
    for i in inds:
        ptr = ptr[i]
    return ptr


def _set_nested(arr, inds, val):
    ptr = arr
    for i in inds[:-1]:
        ptr = ptr[i]
    ptr[inds[-1]] = val


def iter_nested[T: nested|nested_tuple](arr: T, inds: nested|None=None) -> Iterator[tuple[nested, int|T]]:
    """Iterates over inds and values in nested data."""

    if inds is None:
        inds = []
    
    for i, subarr in enumerate(arr):
        inds.append(i)
        subarr = cast(T, subarr)
        yield inds, subarr
        if not isinstance(subarr, int):
            yield from iter_nested(subarr, inds)
        inds.pop()


class Number:
    def __init__(self, arr, verbose=False):
        self.arr = deepcopy(arr)
        self.verbose = verbose

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __getitem__(self, inds):
        return _access_nested(self.arr, inds)

    def __setitem__(self, inds, value):
        _set_nested(self.arr, inds, value)

    def __str__(self):
        return str(self.arr)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        arr = [self.arr, other.arr]
        res = type(self)(arr=arr, verbose=self.verbose)
        res.print(f"After addition: {res}.")
        res.reduce()
        return res

    def ispair(self, inds):
        return isinstance(self[inds], list) and all(isinstance(val, int) for val in self[inds])

    def _it(self, inds=None):
        if inds is None:
            inds = []
        for i in range(len(self[inds])):
            inds_current_level = inds + [i]
            yield inds_current_level
            if isinstance(self[inds_current_level], list):
                for subinds in self._it(inds=inds_current_level):
                    yield subinds
                #
            #
        #
    
    def magnitude(self) -> int:
        res = 0
        for inds, val in iter_nested(self.arr):
            if isinstance(val, int):
                factors = [3 if i == 0 else 2 for i in inds]
                factor = reduce(mul, factors)
                res += factor*val
            #
        return res
    
    def try_explode(self):
        for inds, subarr in iter_nested(self.arr):
            if isinstance(subarr, int):
                continue
            if len(inds) >= EXPLODE_DEPTH and all(isinstance(elem, int) for elem in subarr):
                self.explode(inds)
                return True
            #

        return False

    def split(self, inds):
        self.print(f"Splitting at: {inds}, value={self[inds]}.")
        mid = self[inds] / 2
        a, b = math.floor(mid), math.ceil(mid)
        self[inds] = [a, b]

    def try_split(self):
        for inds, val in iter_nested(self.arr):
            if isinstance(val, int) and val >= SPLIT_THRESHOLD:
                self.split(inds)
                return True
            
        return False

    def _scan_for_number(self, inds, right=True):
        """Scans for the next number. Right and up if rigth = true, otherwise left and up"""

        i = len(inds) - 1

        while i >= 0:
            ind = inds[i]
            if right:
                ind += 1
            else:
                ind -= 1
            
            falloff = ind < 0 or ind >= len(self[inds[:i]])
            
            if falloff:
                i -= 1
                continue
            elif isinstance(self[inds[:i] + [ind]], int):
                return inds[:i]+[ind]
            else:
                subinds = list(self._it(inds[:i]+[ind]))
                if not right:
                    subinds = subinds[::-1]
                for subind in subinds:
                    if isinstance(self[subind], int):
                        return subind

    def explode(self, inds):
        self.print(f"Exploding at: {inds}, value={self[inds]}.")
        # Make sure we're exploding ints only
        a, b = self[inds]
        assert all(isinstance(val, int) for val in (a, b))
        assert len(inds) == 4

        # Replace the 'exploded' pair with a zero
        self[inds] = 0
        # Increment the nearest neighbors with a and b
        for num, right in zip((a, b), (False, True)):
            numinds = self._scan_for_number(inds, right)
            if numinds is not None:
                self[numinds] += num
            #
        #

    def reduce(self):
        done = False
        while not done:
            exploded = self.try_explode()
            if exploded:
                self.print(f"After explode: {self}")
                continue
            else:
                split = self.try_split()
                if split:
                    self.print(f"After split: {self}.")
                    continue
                #
            done = not exploded and not split
        #

    def _get_outer(self, right=True):
        inds = []
        while isinstance(self[inds], list):
            next_ = len(self[inds]) -1 if right else 0
            inds.append(next_)
        return self[inds]


def max_magnitude(*numbers: Number) -> int:
    pairs = ((numbers[i], numbers[j]) for i in range(len(numbers)) for j in range(len(numbers)) if i != j)
    magnitudes = ((a + b).magnitude() for a, b in pairs)
    res = max(magnitudes)
    return res

    for i, a in enumerate(numbers):
        for j, b in enumerate(numbers):
            if i == j:
                continue
            sum_ = a + b
            this_mag = sum_.magnitude()
            if res is None or res < this_mag:
                res = this_mag
            #
        #
    
    if res is None:
        raise RuntimeError
    return res
    


def solve(data: str) -> tuple[int|str, ...]:

    nums = [Number(arr, verbose=False) for arr in parse(data)]

    running = nums[0]
    for num in nums[1:]:
        running += num
    
    star1 = running.magnitude()    
    print(f"Solution to part 1: {star1}")

    star2 = max_magnitude(*nums)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 18
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
