# .·  + `.·  `*·`··.+  `·   +` `·*.·.   ·.`·•  `  · ·  `. *• `·   .·   `+··  .`·
# `·. · *` .·• *·. `       ··+  ` ·  Factory ·· .`       ·`  ·* `   .· ·  .·  · 
# ·`*. ·  `·•   .+ ·   https://adventofcode.com/2025/day/10 ·.   ·  •  .·*·+··.`
# `*·.` ·+*` ·.·    .` · *·   .·    ·`.  +  ·    * ·` ·  .    .`·• ··`. *·`.  +·


import functools
from itertools import combinations
import numba
import numpy as np
from numpy.typing import NDArray
import typing as t
from dataclasses import dataclass

powertype: t.TypeAlias = tuple[int, ...]
buttontype: t.TypeAlias = tuple[tuple[int, ...], ...]

_on = "#"
_off = "."

_light_map = {_on: 1, _off: 0}


@dataclass
class Machine:
    """Represents one of the light machines."""

    indicators: powertype
    buttons: buttontype
    joltage: powertype


def parse(s: str) -> list[Machine]:
    res = []
    for line in s.splitlines():
        parts = line.split(" ")
        indicators = tuple(_light_map[char] for char in parts[0][1:-1])
        buttons = tuple(tuple(map(int, part[1:-1].split(","))) for part in parts[1:-1])
        joltage = tuple(int(c) for c in parts[-1][1:-1].split(","))

        machine = Machine(indicators=indicators, buttons=buttons, joltage=joltage)
        res.append(machine)

    return res


@functools.cache
def brute_force_buttons(buttons: buttontype) -> NDArray[np.int_]:
    """Takes a tuple of buttons, in the format from the input.
    Returns a matrix representing the effect of pressing each combination of buttons, and
    the number of buttons pressed. The number of buttons is stored in the 0'th column.
    The remaining columns store the total increment for each light from pressing the
    combination of buttons."""
    
    # Initialize an array for the result
    dim = 1 + max(val for btn in buttons for val in btn)
    n_rows = 2**len(buttons)
    res = np.zeros(shape=(n_rows, 1+dim), dtype=int)
    
    # Represent the buttons in a matrix
    m_btn = np.zeros(shape=(len(buttons), dim), dtype=int)
    for i, btn in enumerate(buttons):
        for j in btn:
            m_btn[i, j] = 1
        #

    # Compute the total increment to each light from pressing each combination of buttons
    its = ((n_elems, comb) for n_elems in range(len(buttons)+1) for comb in combinations(range(len(buttons)), n_elems))
    for ind, (n_elems, comb) in enumerate(its):
        if not comb:
            continue  # indicing with an empty tuple fails, so skip that
        
        increment = m_btn[np.array(comb)].sum(axis=0)
        res[ind][0] = n_elems
        res[ind][1:] = increment

    return res


@numba.njit(cache=True)
def _match_indicators_numba(indicators: NDArray[np.int_], buttons_brute: NDArray[np.int_]) -> NDArray[np.int_]:
    """Takes an array of indicators, representing the target parity (0 for even, 1 for odd),
    And an array representing the changes produces by each combination of button presses.
    Returns an array representing the button presses which produce the target parity."""

    matching_inds = [i for i in range(0)]
    for i, row in enumerate(buttons_brute):
        toggled = row[1:] % 2
        match = np.all(indicators == toggled)
        if match:
            matching_inds.append(i)
        #
    
    res = buttons_brute[np.array(matching_inds)]

    return res


@functools.cache
def match_indicators(indicators: powertype, buttons: buttontype) -> NDArray[np.int_]:
    """Takes indicators and buttons, and returns a numpy array representing the button presses
    resulting in the specified pattern of indicator lights."""

    m = brute_force_buttons(buttons)
    arr = np.array(indicators)
    res = _match_indicators_numba(arr, m)
    return res


def fewest_toggles(*machines: Machine) -> int:
    """Determines the fewest toggles in which the target pattern of indicator lights
    on the input machine(s) can be attained."""

    res = 0
    for machine in machines:
        m = match_indicators(machine.indicators, machine.buttons)
        # The first column contains the required number of steps, so keep the minimum
        res += min(m[:, 0])

    return res


@functools.cache
def fewest_presses(joltage: tuple[int, ...], buttons: buttontype) -> int:
    """Determines the lowest number of key presses required to attain the input
    joltage, using the input buttons. If no solution can be found, returns -1.
    This works by recursively partitioning the remaining joltage into its possible
    even/odd parity components, then recursing on the even part. Specifically, the state
    of the lights (on/off) at the target joltage is determined, and the number of ways
    to attain that pattern is obtained from a brute force solution. The remainding joltage
    must have even parity, and so we can recurse on half its value."""

    # Zero button presses remain if we've reached the target
    arr = np.array(joltage)
    if all(v == 0 for v in arr):
        return 0
    
    # Find the indicator pattern, and the button combinations resulting in that pattern
    indicators = tuple(arr % 2)
    options = match_indicators(indicators, buttons)

    res = -1
    for row in options:
        # Take the number of button presses, and joltage increase
        n_presses = row[0]
        inc = row[1:]

        # If we overshoot the target, we're on a dead end
        remainder = arr - inc
        if any(v < 0 for v in remainder):
            continue
        
        # Recurse on the even number of button presses
        rec = tuple(remainder // 2)
        sub_sol = fewest_presses(rec, buttons)

        # If there's no solution for the subproblem, we're on a dead end
        if sub_sol < 0:
            continue

        # Add up the odd/even parts
        candidate = n_presses + 2*sub_sol
        if res == -1 or candidate < res:
            res = candidate
        #

    return res


def solve(data: str) -> tuple[int|str, ...]:
    machines = parse(data)

    star1 = fewest_toggles(*machines)
    print(f"Solution to part 1: {star1}")

    star2 = sum(fewest_presses(m.joltage, m.buttons) for m in machines)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 10
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
