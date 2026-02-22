# ·.·`     `·+  .·· `+·*`·     · +`  ·.   ·+ ·.·`  ·  .`*·   ··+.`· ·*`  .·*· `·
# .· ··`.      *  `· · ·.  +·· .* · Crab Cups ·. `·    ··+.`·*.  ·* `·.· ·  ·*.·
# · •.`*`· ·* ·.    ·` https://adventofcode.com/2020/day/23 ·  .   ··`*+·    .·.
#  ·· *·    `·  ` .  ·*` +··    .`   ·*·    · *  . ·`. *·  ·•`  ·. ·  ·* +.·`· *

import numba
import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> NDArray[np.int_]:
    res = np.array([int(digit) for digit in s])
    return res


def reconstruct(arr: NDArray[np.int_], start_at: int=-1, stop_after=-1) -> NDArray[np.int_]:
    """Reconstructs a sequence of cup labels from a linked list.
    arr: Array representing the linked list.
    start_at: The cup label at which the sequence should start. Defaults to the top row of the array.
    stop_after: If specified, stop building the sequence after n steps have been taken after start_at
        has been encountered."""

    shape = len(arr) if stop_after == -1 else stop_after+1
    res = np.full(shape, -1, dtype=int)
    # Find the appropriate place to start iterating
    pointer = 0 if start_at == -1 else next(i for i, val in enumerate(arr[:, 1]) if val == start_at)
    
    for i in range(len(res)):
        val = arr[pointer][1]
        res[i] = val
        pointer = arr[pointer][2]
    
    assert np.all(res != -1)

    return res


@numba.njit(cache=True)
def play_rounds(
        arr: NDArray[np.int_],
        current_cup: int,
        n_rounds: int=1,
        n_pop=3
    ) -> None:
    """Play a number of games of the crab cups game.
    arr: N x 3 array where each row a, b, c contains prev/next pointers (a and c),
        and a value (b). The array represents a linked list, where values can be looked
        up by the row index.
    current cup: The index of the initial current cup.
    n_rounds: Number of rounds to play.
    n_pop: Number of cups to pick up in each round"""

    # Allocate vars for storing the picked up cups, pointers to current+destination cups
    popped_cups = np.full(n_pop, -1, dtype=np.int32)
    destination = -1
    pointer = -1

    for i in range(n_rounds):
        # iterate over next n cups after the current, noting each row index
        pointer = arr[current_cup][2]
        for i in range(n_pop):
            popped_cups[i] = pointer
            pointer = arr[pointer][2]
        
        # Close the gap left by removing a section
        arr[current_cup][2] = pointer
        arr[pointer][0] = current_cup
        
        # Locate the destination cup, where we're going to insert the section
        destination = current_cup - 1
        # Switch cup if out of bounds or if we've already picked up the destination cup
        while destination < 0 or destination in popped_cups:
            destination -= 1
            if destination < 0:
                destination = len(arr) - 1
            #

        # Attach the picked up cups to the destination cup and its clockwise neighbor
        right_of_dest = arr[destination, 2]        
        arr[right_of_dest,0] = popped_cups[-1]
        arr[popped_cups[-1], 2] = right_of_dest
        arr[destination, 2] = popped_cups[0]
        arr[popped_cups[0], 0] = destination

        # Update current cup
        current_cup = arr[current_cup][2]


def crab_cups(cups: NDArray[np.int_], n_rounds=100, n_cups_total: int=-1) -> NDArray[np.int_]:
    """Play a game of crab cups.
    cups: array holding the cups, e.g. [3, 8, 9, ...].
    n_rounds: Number of rounds to play.
    n_cups_total: The desired total number of cups.
        Extends the provided array like [3, 8, 9, ..., 10, 11, ...]"""
    
    labels = cups
    max_cup = max(cups)
    # Extend to the required number of cups
    if n_cups_total != -1:
        labels = np.empty(n_cups_total, dtype=np.int32)
        labels[:len(cups)] = cups
        labels[len(cups):] = np.arange(max_cup+1, max_cup + 1 + (n_cups_total - len(cups)))

    # Make an array where columns 0 and 2 map to previous and next element
    arr = np.arange(len(labels), dtype=np.int32)[:, None].repeat(3, axis=1)
    arr[:, 0] = np.roll(arr[:, 0], +1)
    arr[:, 2] = np.roll(arr[:, 2], -1)
    arr[:, 1] = labels

    # Order by values to allow efficient lookup
    order = np.argsort(arr[:, 1])
    arr = arr[order]
    # Use inverse on the next/prev pointers to retain structure
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    arr[:, 0] = inverse[arr[:, 0]]
    arr[:, 2] = inverse[arr[:, 2]]
    
    # Play the required number of rounds, starting from the current cup
    current_cup = inverse[0]
    play_rounds(arr=arr, current_cup=current_cup, n_rounds=n_rounds)

    return arr


def solve(data: str) -> tuple[int|str, ...]:
    cups = parse(data)
    
    cups_after = crab_cups(cups)
    sequence = reconstruct(cups_after, start_at=1)
    star1 = "".join(map(str, sequence[1:]))
    print(f"Solution to part 1: {star1}")

    all_of_the_cups = crab_cups(cups, n_cups_total=1_000_000, n_rounds=10_000_000)
    sequence2 = reconstruct(all_of_the_cups, start_at=1, stop_after=2)
    star2 = sequence2[1]*sequence2[2]
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
