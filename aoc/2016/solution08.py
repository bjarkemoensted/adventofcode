import numpy as np
import re


def read_input():
    with open("input08.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    """Parses input into a list of tuples like [(command, (arg1, arg2)), ...]"""
    lines = s.split("\n")
    res = []

    regexps = {
        "rect": r"rect (\d+)x(\d+)",
        "rotate row": r"rotate row y=(\d+) by (\d+)",
        "rotate column": r"rotate column x=(\d+) by (\d+)"
    }

    for line in lines:
        for command, pattern in regexps.items():
            if line.startswith(command):
                m = re.match(pattern, line)
                args = list(map(int, m.groups()))
                res.append((command, args))

    return res


def print_display(display):
    lines = [""]
    for row in display:
        chars = ["#" if val > 0 else "." for val in row]
        lines.append("".join(chars))
    lines.append("")

    s = "\n".join(lines)
    print(s)


def run_instructions(instructions):
    display = np.zeros(shape=(6, 50))

    for command, args in instructions:
        a, b = args
        if command == "rect":
            display[:b, :a] += 1
        elif command == "rotate column":
            display[:, a] = np.roll(display[:, a], b)
        elif command == "rotate row":
            display[a, :] = np.roll(display[a, :], b)
        else:
            raise ValueError

    return display


def main():
    raw = read_input()
    instructions = parse(raw)

    final_display = run_instructions(instructions)

    n_pixels = sum(val > 0 for val in final_display.flat)
    print(f"Display contains {n_pixels} pixels that are turned on.")

    print_display(final_display)


if __name__ == '__main__':
    main()
