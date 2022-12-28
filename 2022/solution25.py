def read_input():
    with open("input25.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


test = """1=-0-2
12111
2=0=
21
2=01
111
20012
112
1=-1=
1-12
12
1=
122"""


digitmap = {
    "0": 0, "1": 1, "2": 2, "-": -1, "=": -2
}


def parse(s):
    res = s.split("\n")
    return res


def decimal_from_snafu(snafu_number):
    pow = 0
    res = 0
    for char in snafu_number[::-1]:
        n = digitmap[char]
        res += n*5**pow
        pow += 1

    return res


def snafu_from_decimal(decimal):
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


def main():
    raw = read_input()
    snafu_input = parse(raw)
    target = sum(decimal_from_snafu(num) for num in snafu_input)

    target_snafu = snafu_from_decimal(target)
    print(f"The correct input number is: {target_snafu}.")


if __name__ == '__main__':
    main()
