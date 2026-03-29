# .`.···` *   .    ·*`   . * ··`· .•    `·*.   ·*   ·.   + .`· *·      . ` . +*·
# · `.   `·*·  ·    ` .     ·.* Binary Diagnostic *· * · `  ·.*• ·`      ··  ·.`
# `·+   *  · *   .·    https://adventofcode.com/2021/day/3 ·.+` * ·`   · .  ` ·.
# **·  . ·.*  ·  +` .*· .    `·*   . ·`   `  +*. · `*·.   · + ·. `•   ·.·   * `·

from collections import Counter
from typing import Callable, Sequence


def parse(s: str) -> list[str]:
    return s.splitlines()


def search_with_bit_criteria(candidates: Sequence[str], criterion: Callable, ind=0) -> str:
    """Recursively eliminates candidate lines from the data based on whether they
    meet the input criterion. Returns the result when only one candidate remains."""

    if len(candidates) == 1:
        return "".join(candidates[0])
    # Identify target bit
    bit_counts = Counter([line[ind] for line in candidates])
    if bit_counts["0"] == bit_counts["1"]:
        # For ties, use 0 if criterion = min and 1 if criterion = max
        target_bit = criterion(bit_counts.keys())
    else:
        target_bit = criterion(bit_counts.items(), key=lambda t: t[1])[0]

    # Update candidate set
    candidates = [line for line in candidates if line[ind] == target_bit]
    # Recursive step - apply criterion to the next bit
    res = search_with_bit_criteria(candidates, criterion, ind + 1)
    return res


def determine_gamma(*diagnostics: str) -> str:
    """Determines gamma - the most common bit at each position"""
    len_ = len(diagnostics[0])
    assert all(len(d) == len_ for d in diagnostics)

    res = ""
    for bit_pos in range(len_):
        bits = (d[bit_pos] for d in diagnostics)
        most_common = max(Counter(bits).items(), key=lambda t: t[1])[0]
        res += str(most_common)

    return res

def solve(data: str) -> tuple[int|str, ...]:
    diagnostics = parse(data)
    gamma = determine_gamma(*diagnostics)
    # Take the negation
    epsilon = "".join([{"0": "1", "1": "0"}[bit] for bit in gamma])
    
    star1 = int(gamma, 2) * int(epsilon, 2)
    print(f"Solution to part 1: {star1}")

    lines = [line for line in diagnostics]
    oxygen_generator_rating = search_with_bit_criteria(lines, criterion=max)
    c02_scrupper_rating = search_with_bit_criteria(lines, criterion=min)
    star2 = int(oxygen_generator_rating, 2) * int(c02_scrupper_rating, 2)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
