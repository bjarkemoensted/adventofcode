# · * ·+.`·         ·· + .·*`  ·   · ·.+ ·   `·* ·  `*·  ·.   .·`* ·   ·`+ ·` ·`
# `·  *`  .·•· ·  *  .*·  Flawed Frequency Transmission   · •·  · +`·*   ·. ·• ·
# ··` ` ·*  ·*   · +`  https://adventofcode.com/2019/day/16 · * .`·+ ·   .`* ·`+
# .`·.·  · •.     ··*  . ·· ·+  •·`.· `. + ·    *. · . •+   ·`·.*      *· ·`* ·.

import numpy as np
from numba import njit
from numpy.typing import NDArray

BASE_PATTERN = np.array([0, 1, 0, -1])


def parse(s: str) -> NDArray[np.int_]:
    return np.array([int(digit) for digit in s])


@njit(cache=True)
def fft(x: NDArray[np.int_], n_phases: int) -> NDArray[np.int_]:
    """Runs the full FFT procedure on the input signal, n times."""
    
    n_vals = len(x)
    x = x.copy()
    temp = x.copy()  # holds signal after current iteration

    for _ in range(n_phases):
        for i in range(n_vals):
            running = 0
            n_reps = i+1  # n repetitions
            n_used = 1  # skip first
            # current element of the base pattern
            ind = 0

            for j in range(n_vals):
                if n_used == n_reps:
                    n_used = 0
                    ind += 1
                    # Cycle values from the base pattern
                    if ind >= len(BASE_PATTERN):
                        ind = 0
                    #
                
                running += x[j]*BASE_PATTERN[ind]
                n_used += 1

            # Absolute values mod 10
            if running < 0:
                running = -running
            temp[i] = running % 10
        
        # Swap the signal and temporary array
        x, temp = temp, x
    return x


@njit(cache=True)
def _fill(x: NDArray[np.int_], arr: NDArray[np.int16], message_offset: int, n_phases: int):
    """Fill in the values for each phase, assuming we're only looking at values in the latter
    half of the signal, where the FFT transformation matrix is triangular.
    In this case, a recurrence relation exists where the value of x_i^n (the i'th element of
    the signal vector after n FFT steps) is
    x_i^n = x_i^(n-1) + x_(i+1)^n"""
    
    # The last element of the vector never changes
    arr[-1, :] = x[-1]
    # Set all initial values (n=0) to the initial signal
    arr[:, 0] = x.copy()

    for i in range(len(x)-2, message_offset-1, -1):
        for j in range(1, n_phases+1):
            # Use recurrence relation to fill in missing values
            arr[i, j] = (arr[i+1, j] + arr[i, j-1]) % 10
        #
    #


def clean_signal(
        x: NDArray[np.int_],
        message_offset: int=0,
        n_phases=100,
        message_length=8,
    ) -> str:
    """Uses FFT to clean a signal and return the resulting message.
    If the message offset is greater than half the signal length, a trick can be use
    to compute the result much more efficiently, by expressing each element of the
    signal vector as a function of the next, and using a recurrence to fill in the
    values for each of the n FFT steps."""
    
    if message_offset >= len(x) // 2:
        # Use the trick. Initialize the array, then use an optimized numba function to populate it
        arr = np.empty(shape=(len(x), n_phases+1), dtype=np.int16)
        _fill(x=x, arr=arr, message_offset=message_offset, n_phases=n_phases)
        code = arr[message_offset:message_offset+message_length, n_phases]
    else:
        # Otherwise, run the full procedure
        code = fft(x, n_phases=n_phases)
        code = code[message_offset: message_offset+message_length]
    
    # Convert into string and return
    res = "".join(map(str, code))
    return res


def solve(data: str) -> tuple[int|str, ...]:
    signal = parse(data)
    
    star1 = clean_signal(signal)
    print(f"Solution to part 1: {star1}")

    signal = np.tile(signal, reps=10_000)
    message_offset = int("".join(map(str, signal[:7])))
    star2 = clean_signal(
        signal,
        message_offset=message_offset,
        message_length=8,
        n_phases=100
    )

    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 16
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
