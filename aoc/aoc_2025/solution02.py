# .`  ·+··` . · *.  ·. * · • ` .  · ·+  ·  · .   ·    `·.· *·     * `.  +···` ·.
# *. ·· +`· *   . ·  `· · ..* · `·  Gift Shop ·  `·* .+`  ·     *·.   `*·    `.·
# `··.   ·• `·*. *`· · https://adventofcode.com/2025/day/2 · +··    .·  .`+·· *.
# ··•*.`·     ·*·  .  •.·` ·+ * ·   .·`·+.   ·. ·* . ·* `.   ·*`· · ·* ·  . *·``

import string
import typing as t
from functools import cache
from itertools import product


def parse(s: str) -> list[tuple[int, int]]:
    res: list[tuple[int, int]] = []
    for part in s.split(","):
        a, b = part.split("-")
        res.append((int(a), int(b)))
    
    return res


@cache
def _repeating_digits(n_digits: int) -> tuple[str, ...]:
    """Returns all n-digit patterns, excluding leading zeroes"""
    _digits = string.digits
    patterns = ("".join(digits) for digits in product(_digits, repeat=n_digits))
    return tuple(patterns)


@cache
def _repeating_patterns(pattern_length: int, n_repititions: int) -> tuple[int, ...]:
    """Returns all repeating patterns with the specified pattern length and number of pattern repititions"""
    patterns = _repeating_digits(pattern_length)
    res = tuple(int(n_repititions*pattern) for pattern in patterns if not pattern.startswith("0"))
    return res


def _generate_fixed_pattern_length(a: int, b: int, pattern_length: int, n_repititions: int) -> t.Iterator[int]:
    """Generate all numbers consisting of the given pattern length and repititions, in the (a, b) interval,
    both inclusive."""

    assert b >= a
    numbers = _repeating_patterns(pattern_length=pattern_length, n_repititions=n_repititions)

    for number in numbers:
        if number < a:
            continue
        if number > b:
            return
        yield number


def _generate_invalid_fixed_pattern_length(range_: tuple[int, int], n_repititions: int) -> t.Iterator[int]:
    """Generates all numbers which 1) are inside the input range, and 2) consist of n repititions
    of a pattern"""

    # Determine the number of digits we might match in the range
    low, high = (len(str(lim)) for lim in range_)
    n_digits_usable = (n_digits for n_digits in range(low, high+1) if n_digits % n_repititions == 0)

    # For every n_digits in the range, generate the invalid codes with n_repititions
    for n_digits in n_digits_usable:
        pattern_len = n_digits // n_repititions
        yield from _generate_fixed_pattern_length(*range_, pattern_len, n_repititions)


def compute_invalid_codes(ranges: list[tuple[int, int]], n_repititions: int|None=None) -> tuple[int, ...]:
    """Returns a tuple of all invalid codes in the specified ranges, using the input number
    of repititions of patterns for the numbers. If not specified, all n_repititions of at least 2 are used."""

    # Store invalid codes in a set to avoid double counting ('111111' can be both 2x'111' and 3x'11', for instance)
    invalid = set([])
    
    for range_ in ranges:        
        reps = list(range(2, len(str(max(range_))) + 1)) if n_repititions is None else [n_repititions]
        for rep in reps:
            invalid |= set(_generate_invalid_fixed_pattern_length(range_, n_repititions=rep))
    return tuple(sorted(invalid))


def solve(data: str) -> tuple[int|str, ...]:
    ranges = parse(data)
    
    star1 = sum(compute_invalid_codes(ranges, 2))
    print(f"Solution to part 1: {star1}")

    star2 = sum(compute_invalid_codes(ranges))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 2
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
