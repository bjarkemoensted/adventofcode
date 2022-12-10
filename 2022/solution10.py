def read_input():
    with open("input10.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    """Parse input into tuples like (addx, 42) or (noop, None)"""
    instructions = []
    for line in s.split("\n"):
        stuff = line.split(" ")
        if len(stuff) == 1:
            instruction = (line, None)
        else:
            a, b = line.split(" ")
            instruction = (a, int(b))
        instructions.append(instruction)

    return instructions


def run_instructions(instructions, starting_value=1):
    """Runs all instructions and keeps the corresponding register values"""

    X = [starting_value]
    for operation, value in instructions:
        # noop does nothing, so keep the preceding value
        if operation == "noop":
            X.append(X[-1])
        # addx spends 2 cycles updating the value
        elif operation == "addx":
            X.append(X[-1])
            X.append(X[-1] + value)
        else:
            raise ValueError

    # we've been storing the value in the upcoming cycle, so drop the final value again
    return X[:-1]


def compute_signal_strength(register_values):
    """Sums register value*cycle number for some particular cycles."""
    interesting_inds = [20, 60, 100, 140, 180, 220]
    # Use zero indexing, so subtract one from the cycle numbers to get the cycle index
    strength = sum(i*register_values[i-1] for i in interesting_inds)
    return strength


def coord_from_cycle_number(cycle_number, display_width_pixels=40):
    """Get the row+column numbers for a given pixel number, on a display
    with the specified width."""
    i = cycle_number // display_width_pixels
    j = cycle_number % display_width_pixels
    return i, j


def draw_the_thing(arr, width=40):
    """Returns a string representation of the display output."""

    # Initially set all pixels off ('.')
    cols = len(arr) // width
    pixel_values = [["." for _ in range(width)] for _ in range(cols)]

    for cycle, x in enumerate(arr):
        i, j = coord_from_cycle_number(cycle)
        # Draw pixel if x overlaps with the sprite at x +/- 1
        if abs(j - x) <= 1:
            pixel_values[i][j] = "#"

    res = "\n".join(["".join(row) for row in pixel_values])
    return res


def main():
    raw = read_input()
    instructions = parse(raw)

    arr = run_instructions(instructions)
    signal = compute_signal_strength(arr)
    print(f"The signal strength is {signal}.\n")

    s = draw_the_thing(arr)
    print(s)


if __name__ == '__main__':
    main()
