#  .·  * .`·  .·· +` `· *  . ·      `*··  .+·*`·  .  · . ·   ·`.     ·*.· · .·.*
# ·*  `.··   ·.  `  · *.·    Grove Positioning System *·   ` . · ·  ·* ·` +.  ·.
# `·.·    .* `·  .· .· https://adventofcode.com/2022/day/20  * `·.· + ·.  ·  `· 
# .`* ·· * `.* ··+ · . ·`·   + ·  ·     .··  `.    *·*. ·   ··.  ·* .  ·.* `·+ ·

from typing import Self

import numpy as np
from numba import njit
from numpy.typing import NDArray

DECRYPTION_KEY = 811589153
GROVE_COORD_INDS = (1000, 2000, 3000)
dtype = np.int64


def parse(s: str) -> NDArray[np.int_]:
    return np.array(list(map(int, s.splitlines())), dtype=dtype)


@njit
def mix_arr(arr: NDArray[np.int_], row: int) -> None:
    """Performs the mixing operation in-place on the specified row in the array.
    This works by first detaching the node at the input index, then stepping n steps along the
    linked list and inserting the node there."""
    
    val = arr[row, 1]
    if val == 0:
        return  # Nothing to do if moving the node zero places
    
    # Determine the direction (go via 'previous' for negative values, otherwise 'next')
    prevind, nextind = ((0, -1) if val > 0 else (-1, 0))
    # Grab the pointers to neighbor nodes before detaching
    oldprev = arr[row, prevind]
    oldnext = arr[row, nextind]

    # Detach the node from its current neighbors
    arr[oldprev, nextind] = oldnext
    arr[oldnext, prevind] = oldprev
    
    # Not necessary, but ensures consitency by making the detached node its own tiny linked list
    arr[row, nextind] = row
    arr[row, prevind] = row

    target = oldnext
    # Number of steps to take. Using modulo of the LL after detaching the node, to avoid looping
    n_steps = (abs(val) - 1) % (len(arr) - 1)
    for _ in range(n_steps):
        target = arr[target, nextind]
        if target == row:
            raise ValueError
        #

    # Attach the node at the target position.
    arr[row, prevind] = target
    arr[row, nextind] = arr[target, nextind]
    # Update neighbor pointers
    arr[arr[target, nextind], prevind] = row
    arr[target, nextind] = row


class LinkedList:
    """A linked list holding integers.
    Represents the linked list with N integers as an N x 3 numpy array. The center column holds the values,
    and the left/right columns hold pointers to the row representing the previous/next node in the list."""

    def __init__(self, numbers: NDArray[np.int_]) -> None:
        self.arr = np.arange(len(numbers), dtype=dtype)[:, None].repeat(3, axis=1)
        self.arr[:, 0] = np.roll(self.arr[:, 0], +1)
        self.arr[:, 1] = numbers.copy()
        self.arr[:, 2] = np.roll(self.arr[:, 2], -1)
    
    def as_list(self) -> list[int]:
        """Convert into a list of values"""
        current = 0
        res = []
        for _ in range(len(self.arr)):
            res.append(self.arr[current, 1].item())
            current = self.arr[current, -1]
        return res

    def mix(self, n=1) -> Self:
        """Performs the mixing operation, optionally multiple times. Returns self to allow chaining."""
        for _ in range(n):
            for i in range(len(self.arr)):
                mix_arr(self.arr, i)
            #
        return self
    
    def grove_coords_sum(self) -> int:
        """Computes the sum of the grove coordinates"""
        res = 0
        numbers = self.as_list()
        # Start the list from the 0 position
        offset = numbers.index(0)
        for crd in GROVE_COORD_INDS:
            ind = (offset + crd) % len(numbers)
            res += numbers[ind]
        
        return res
    #


def solve(data: str) -> tuple[int|str, ...]:
    numbers = parse(data)

    star1 = LinkedList(numbers).mix().grove_coords_sum()
    print(f"Solution to part 1: {star1}")

    star2 = LinkedList(DECRYPTION_KEY*numbers).mix(10).grove_coords_sum()
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 20
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
