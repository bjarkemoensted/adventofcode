# • .·· ·.`  ··.   *·`  *· .  ·`  ·   ·+.`       .*.·`·   ` • ·* .· ` ·.`·* .· ·
# ·  •.·`•  * `·.·  .·+   `*·  · Trash Compactor *··    `·.*·    * .·`  +  ·`  .
# . ·+*` ·. · .  `· *  https://adventofcode.com/2025/day/6  .  ·`·.  ·.`  ·   ·`
# ·· .`·     `+ ·.* `.·.·* ·` +.  `·*  ·•.` ·  * ·  .* ·.` · ·  .  `  *··  .* `·

import typing as t
from dataclasses import dataclass
from functools import reduce
from operator import add, mul

ops = {
    "+": add,
    "*": mul
}


@dataclass(init=False)
class Problem:
    """Data class for storing details of a math problem.
    This just stores the operator symbol and the numbers represented as strings,
    and exposes methods for getting the corresponding mathematical operator, and
    iterating over the numbers (both by rows and by digits/columns)"""

    operator_symbol: str
    number_strings: tuple[str, ...]

    def __init__(self, operator_symbol: str, *number_strings: str) -> None:
        """Inits a problem. operator_symbol is the symbol '+'/'*' of the operator to use
        for the problem. number_strings is the numbers involved, represented as string."""

        # Assign the operator and numbers
        self.operator_symbol = operator_symbol
        self.op: t.Callable[[int, int], int] = ops[self.operator_symbol]
        self.number_strings = tuple(number_strings)

        # Determine the max number of digits
        n_digits = list({len(ns) for ns in self.number_strings})
        assert len(n_digits) == 1
        self.n_digits: int = n_digits[0]

    def numbers_by_row(self) -> t.Iterator[int]:
        """Generate the numbers in the problem (top row, second row, etc)"""
        yield from map(int, self.number_strings)

    def numbers_by_digits(self) -> t.Iterator[int]:
        """Generate numbers by digits (first all rightmost digits, then all digits in
        the second rightmost position, etc)."""

        for i in reversed(range(self.n_digits)):
            numstring = "".join(n[i] for n in self.number_strings)
            yield int(numstring)
        #
    #


def parse(s: str) -> list[Problem]:
    res = []
    cells = [list(line) for line in s.splitlines()]
    ncols = len(cells[0])
    assert all(len(row) == ncols for row in cells)

    # Determine the column separator, where every row contains a blank space
    cuts = [j for j in range(ncols) if all(row[j] == " " for row in cells)]
    cuts = cuts + [ncols]  # also 'cut' at the rightmost edge
    
    # Parse each column into a problem instance
    left = 0
    for right in cuts:
        column = ["".join(row[left:right]) for row in cells]
        # Extract the operator and numbers, and instantiate the problem
        operator_symbol = column[-1].strip()
        numstrings = column[:-1]
        p = Problem(operator_symbol, *numstrings)
        
        # Proceed to next column
        res.append(p)
        left = right+1
    
    return res


def check_math(*problems: Problem, by_row=True) -> int:
    """Do the math thing.
    problems: Problem dataclass instances, representing each individual problem.
    by_row (bool, default True) indicates whether to consider each row a number, as opposed
    to iterating by digit."""

    res = 0
    for p in problems:
        numbers = p.numbers_by_row() if by_row else p.numbers_by_digits()
        res += reduce(p.op, numbers)
    
    return res


def solve(data: str) -> tuple[int|str, ...]:
    problems = parse(data)

    star1 = check_math(*problems)
    print(f"Solution to part 1: {star1}")

    star2 = check_math(*problems, by_row=False)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 6
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
