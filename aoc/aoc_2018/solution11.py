# `.·. *·   ·   ·+`. ·  * ··.` · .   `  +··*   .` · .`  ·* •.·`*   ·` *  . · ·.*
# *·. .  ` ·   + .* ·`     .  .·  Chronal Charge  .·   . ·     .· `*· ·.  ·`*.·.
# . *·`·   * ` . `· .  https://adventofcode.com/2018/day/11  `*·+.·      ·  `.+·
# ·`` •. ·.   ·    · .* ·   ` ·+. · `   ·.·.*·     * ·  .· `  .. ·.`+· · *·+.·` 

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def parse(s: str):
    return int(s)


def make_power_square(serial_number: int) -> np.typing.NDArray[np.int_]:
    """Compute an array containing the power at each coordinate"""
    
    size = 300
    # Make arrays of coordinates (zero-indexed!)
    y_coords, x_coords = np.indices((size, size))
    # Follow the problem instructions to get the power at each coordinate
    rack_id = x_coords + 1 + 10
    power = rack_id*(y_coords+1)
    power += serial_number
    power *= rack_id
    power = (power % 1000) // 100
    power -= 5
    
    return power


def find_max_power(power_grid, size=3) -> tuple[int, int]:
    """Finds the coordinates (top left corner) of the square with the input size with the
    greatest possible sum of power within its area."""
    
    # Make a sliding N x N window over the power and sum it
    windows = sliding_window_view(power_grid, (size, size))
    result = windows.sum(axis=(-2, -1))
    
    # Get the index of the max
    ind = tuple(int(val) for val in np.unravel_index(np.argmax(result), result.shape))
    assert len(ind) == 2

    return ind


def square_with_largest_intensity(power: np.typing.NDArray[np.int_]) -> tuple[int, int, int]:
    """Determines the square (of any size) with the greatest total intensity, defined as the sum
    of all cells within the square. Optimized by using the summed-area table algorithm."""
    
    h, w = power.shape
    assert h == w
    
    # compute the summed-area-table, which can be used to efficiently compute intensities
    sat = np.zeros(shape=(h+1, w+1), dtype=np.int_)
    for i, j in np.ndindex(power.shape):
        sat[i+1, j+1] = power[i, j] + sat[i+1, j] + sat[i, j+1] - sat[i, j]
    
    # Initialize a 3D array with value a i, j, k representing intensity of square size k at i, j. 
    sentinel = -1
    intensities = np.full((h, h, h), sentinel, dtype=np.int32)
    
    # Populate the intensity array
    for n in range(1, h+1):
        valid_h = h - n + 1
        valid_w = w - n + 1
        top_left     = sat[:valid_h, :valid_w]
        bottom_right = sat[n:, n:]
        top_right    = sat[:valid_h, n:]
        bottom_left  = sat[n:, :valid_w]

        intensity = bottom_right + top_left - top_right - bottom_left
        intensities[:valid_h, :valid_w, n - 1] = intensity
    
    # Get the index of the max
    res = tuple(int(val) for val in np.unravel_index(intensities.argmax(), intensities.shape))
    # If we used a sentinel greater than the true max, we'd get an error, so just check to make sure
    assert intensities[*res] != sentinel
    assert len(res) == 3  # this is just to help the type checker

    return res


def solve(data: str) -> tuple[int|str, int|str]:
    serial_number = parse(data)
    power = make_power_square(serial_number)
    
    i, j = find_max_power(power, size=3)
    star1 = f"{j+1},{i+1}"
    print(f"Solution to part 1: {star1}")
    
    ia, ja, size = square_with_largest_intensity(power)
    star2 = f"{ja+1},{ia+1},{size+1}"
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 11
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
