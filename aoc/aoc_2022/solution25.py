# ·.` *.`·*  ·   .· ·* · ` · `* .·· .  `* .·    ·   . ··`    * .· ·` · * .  `.··
# `+.··*   · . · *`+. ·  `·* ··  Full of Hot Air · .    ·*  · *·.·    `  · .+·`.
# ·` +· .`·.   `·`*·   https://adventofcode.com/2022/day/25   ·   .·• +`· ·*  ·`
# .·*`. · .`·*       ·  `· .* `  ·.  `· `  ··.*· ` • ·   ··+. `   ·*. ·  `.·· .+


def parse(s: str) -> list[str]:
    return s.splitlines()


digitmap = {
    "0": 0, "1": 1, "2": 2, "-": -1, "=": -2
}


def decimal_from_snafu(snafu_number: str) -> int:
    pow = 0
    res = 0
    for char in snafu_number[::-1]:
        n = digitmap[char]
        res += n*5**pow
        pow += 1

    return res


def snafu_from_decimal(decimal: int) -> str:
    """Converts a decimal number to SNAFU format."""

    # Start by finding the lowest power that allows to overshoot the specified decimal number
    pow_ = 0
    while sum(3*5**n for n in range(pow_)) < decimal:
        pow_ += 1

    # Make the largest possible N-digit number
    running = [3 for _ in range(pow_)]

    # Iterate over digits, starting from the most significant one
    for i in range(len(running)-1, -1, -1):
        # This is how much we can subtract by decreasing the n-1 smaller digits
        can_subtract = sum(5*5**n for n, val in enumerate(running[:i]))

        # While we're off by more than the remaining digits can compensate for, reduce the most significant digit
        while sum(val*5**n for n, val in enumerate(running)) - decimal > can_subtract:
            running[i] -= 1

    # Convert to a string representation of the number in the SNAFU base
    val2digit = {v: k for k, v in digitmap.items()}
    digits = []
    for val in running[::-1]:
        digit = val2digit[val]
        digits.append(digit)

    res = "".join(digits)
    return res


def solve(data: str) -> tuple[int|str|None, ...]:
    snafu_input = parse(data)
    target = sum(decimal_from_snafu(num) for num in snafu_input)

    star1 = snafu_from_decimal(target)
    print(f"Solution to part 1: {star1}")

    star2 = None

    return star1, star2


def main() -> None:
    year, day = 2022, 25
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
