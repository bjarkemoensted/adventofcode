# ·*.   ·.*  ·`*  . +··     *·  · `*    •·*·`.  *  · . `*+`·  .*  ·   ·`·**  `· 
# *` * ·`  ·*    .  ·`       .· Monitoring Station     . · • `* · .*·     ·.+* `
# `· `·*+ ·   .   ·* • https://adventofcode.com/2019/day/10 ··`   *  · + * `· .·
# . ··*`  .  .*   +·   * ·.·    .* ` +·     .+ ·`  *·  *  · `+·. * · ` ·   +`.**

import numpy as np
from numpy.typing import NDArray


def parse(s: str) -> NDArray[np.str_]:
    res = np.array([list(line) for line in s.splitlines()])
    return res


def scale(arr: NDArray[np.int_]) -> NDArray[np.int_]:
    """Scales each row in the array by its greatest common divisor.
    This is so e.g. coordinates (4, 3) and (8, 6) get the same values."""

    gcds = np.gcd(arr[:, 0], arr[:, 1])
    scaled = np.zeros_like(arr)
    np.floor_divide(arr, gcds[:, None], out=scaled, where=arr != 0)
    return scaled


def count_asteroids_in_los(asteroids: NDArray[np.int_], position: NDArray[np.int_]) -> int:
    """Takes an array with asteroid positions (2D array), and the
    coordinates of an asteroid monitoring station.
    Returns the number of asteroids to which the station has a direct
    line of sight"""

    diffs = asteroids - position
    scaled = scale(diffs)

    unique = np.unique(scaled, axis=0)
    n, _ = unique.shape
    res = n - 1
    return res


def pewpewpew(asteroids: NDArray[np.int_], position: NDArray[np.int_]) -> NDArray[np.int_]:
    """Determines the order in which the asteroids are blown up by a laser which initially
    shoots directly upwards from the specified position, then rotates in the positive direction,
    only destroying a single asteroid in each distinct line of sight in each round."""
    
    # Compute distances and scale by gcd
    targets = asteroids[~np.all(asteroids == position, axis=1)]
    diffs = targets - position
    scaled = scale(diffs)

    # Determine order based on (angle, distance)
    dists = np.abs(diffs).sum(axis=1)
    angles = np.arctan2(scaled[:, 0], scaled[:, 1])
    # Shift by 90 degrees because the laser starts in the y, not x, direction
    degrees = (np.degrees(angles) + 90) % 360
    
    # Assign a 'score' to each asteroid, denoting when the laser will hit during a 360 deg sweep
    sweep = np.lexsort((dists, degrees))
    scores = np.zeros_like(sweep, dtype=float)
    # Keep scores in the [0, 1) interval
    scores[sweep] = np.linspace(0, 1, len(sweep), endpoint=False)

    # Count the number of asteroids along each distinct line-of-sight
    prev: tuple[int, int]|None = None
    depth = 0
    for ind in sweep:
        # Reset when the line-of-sight changes
        y, x = scaled[ind]
        new_los = prev is None or (y, x) != prev
        if new_los:
            depth = 0
            prev = y, x
        # Increment depth. These are multiples of 1, so will dominate the sweep order
        scores[ind] += depth
        depth += 1
    
    # Sort the asteroids according to their order of destruction
    order = np.argsort(scores)
    alt = targets[order]
    return alt


def solve(data: str) -> tuple[int|str, ...]:
    map_ = parse(data)
    asteroids = np.argwhere(map_ == "#")
    pos = max(asteroids, key=lambda p: count_asteroids_in_los(asteroids, p))
    star1 = count_asteroids_in_los(asteroids, pos)
    print(f"Solution to part 1: {star1}")

    asteroids = np.vstack([asteroids, pos])
    hmm = pewpewpew(asteroids, pos)
    y, x = hmm[199]
    star2 = 100*x + y
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 10
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
