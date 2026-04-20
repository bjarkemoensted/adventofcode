# `·.  *·+   `*·   *· * .   ·+·+    `·.*·.  +·`*      ·`+.·  ·`• ·*·   .*  `·.··
# *.· `·   · + `·.*    ·  ·`* ` Regolith Reservoir ·  + ·`  ·* .·+ *   · `*·. `·
# ·`*`· .·     ·   ·   https://adventofcode.com/2022/day/14 .•+·  ·`  ·  .·+`·.`
# *· .  · .*  · +*·.`· ` *·    ·+*· .   `      .*· *·` ·  . · ·* .` · * +· . `*·

import numpy as np
from numba import njit
from numpy.typing import NDArray

_ON = "#"
_OFF = "."


def parse(s: str) -> list[list[tuple[int, int]]]:
    paths = []
    for line in s.splitlines():
        parts = [part.split(",") for part in line.split(" -> ")]
        paths.append([(int(a), int(b)) for a, b in parts])
    
    return paths


def display(arr: NDArray[np.bool_]) -> None:
    """Displays a sand array as ASCII, for debugging etc"""
    i, j = np.where(arr)
    imin, imax = i.min(), i.max()
    jmin, jmax = j.min(), j.max()
    rows, cols = arr.shape

    eps = 3
    M = arr[max(0, imin - eps): min(rows, imax + eps), max(0, jmin - eps): min(cols, jmax + eps)]
    
    d = {True: _ON, False: _OFF}
    s = "\n".join(("".join(row) for row in np.vectorize(d.get)(M)))
    print(s)


def make_array(paths: list[list[tuple[int, int]]]) -> NDArray[np.bool_]:
    """Takes a list of paths (each a list of tuples of x, y coordinates) and returns
    a boolean array with the points on the paths set to True and False elsewhere."""
    
    # Determine ranges of x and y needed
    x, y = zip(*sum(paths, []))
    assert all(all(v >= 0 for v in vals) for vals in (x, y))
    x_max = max(x)
    y_max = max(y)

    arr = np.full(shape=(y_max+1, x_max+1), fill_value=False, dtype=bool)

    for path in paths:
        for ind in range(len(path)-1):
            # Get subsequent pairs of points and determine the steps between them
            p1 = np.array(path[ind])
            p2 = np.array(path[ind+1])
            delta = p2 - p1
            assert sum(v != 0 for v in delta) == 1
            n_steps = sum(np.abs(delta))
            step = delta // n_steps
            assert np.all((p1 + n_steps*step) == p2)

            # Fill into the array
            for n in range(n_steps+1):
                x, y = p1 + n*step
                arr[y, x] = True
            #
        #
    
    return arr


@njit(cache=True)
def _trace_grain(arr: NDArray[np.bool_], i, j, directions: NDArray[np.int_]) -> bool:
    """Follows a single grain of sand through the array, until it comes to rest.
    Attempts each the input directions in order, coming to rest if all fail.
    Returns a boolean indicating whether the grain could fall at all."""
    
    # False if there's no room at the initial location
    if arr[i, j]:
        return False
    
    rows, cols = arr.shape

    while True:
        moved = False
        for di, dj in directions:
            ip, jp = i+di, j+dj
            # Done if the grain falls out of bounds
            if jp < 0 or ip >= rows or jp >= cols:
                return False
            # Successfully moved if not out of bounds and no wall
            if not arr[ip, jp]:
                i, j = i+di, j+dj
                moved = True
                break
            #
        # The grain comes to rest if no move was possible
        if not moved:
            arr[i, j] = True
            return True
        #
    #


def trickle(arr: NDArray[np.bool_], x0=500) -> int:
    """Allows sand to trickle from the specified x-position (from y=0), until
    no further grains can fall. Returns the final number of sand grains"""

    arr = arr.copy()
    n_grains = 0
    # Direction order - down, down+left, down+right
    directions = np.array([[1, 0], [1, -1], [1, 1]])

    # Repeatedly update the array with the next grain, until out of space
    while True:
        landed = _trace_grain(arr, i=0, j=x0, directions=directions)
        if not landed:
            break
        n_grains += 1

    return n_grains


def expand_array(arr: NDArray[np.bool_], x0=500, delta_y=2) -> NDArray[np.bool_]:
    """Expands the array with a 'floow' where sand may land. Expands horizontally to have
    enough room even for sand grains that take a completely diagonal trajectory
    from the input x0."""

    # Determine new shape
    arr = arr.copy()
    rows, cols = arr.shape
    imax_new = rows + delta_y
    jmax_new = x0 + imax_new

    # Pad the bottom of the array
    lower = np.full(shape=(delta_y, cols), fill_value=False, dtype=bool)
    arr_expand = np.vstack([arr, lower])

    # Pad array to the right
    n_pad_right = jmax_new - cols
    if n_pad_right > 0:
        padding = np.full(shape=(imax_new, n_pad_right), fill_value=False, dtype=bool)
        arr_expand = np.hstack([arr_expand, padding])

    # Ensure there's sufficient space to the left
    assert x0 - imax_new > 0
    
    # Add the floor
    arr_expand[-1, :] = True

    return arr_expand


def solve(data: str) -> tuple[int|str, ...]:
    paths = parse(data)
    M = make_array(paths)

    star1 = trickle(M)
    print(f"Solution to part 1: {star1}")

    M2 = expand_array(M)

    star2 = trickle(M2)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 14
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
