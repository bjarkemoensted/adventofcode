# +.*⸳ꞏ `*.⸳  • ⸳` ꞏ.      ⸳.+ ⸳*    `.•⸳  `•.ꞏ   ⸳ `.  `⸳`⸳* + .` *  ꞏ⸳    ꞏ .+
# ꞏ+⸳`    ⸳•*`  .⸳.`*   ` ⸳ Two-Factor Authentication ⸳`  ⸳  • ꞏ . `* ⸳   + .`⸳ꞏ
# • ꞏ ⸳+ .  `  ⸳ꞏ*     https://adventofcode.com/2016/day/8      `    ⸳ `  ⸳  • .
#  ⸳ *`⸳ꞏ ` ꞏ⸳.   *⸳  ꞏ .   ⸳`*   . ⸳ • + +`⸳  ꞏ.  ꞏ*   •   ` ꞏ  • ⸳.`. *ꞏ  • `⸳


from aococr import aococr
import numpy as np
import re


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


def solve(data: str):
    instructions = parse(data)

    final_display = run_instructions(instructions)

    star1 = sum(val > 0 for val in final_display.flat)
    print(f"Solution to part 1: {star1}")

    star2 = aococr(final_display)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2016, 8
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
