#  ·`•`  ·.+·  .·   · · `.   *  `.· *.  ·+ ` . ·    .·` · .+  · ` •. ·    `·  .·
# + ·`•.·* .  `  ·. *   .·`·  Probably a Fire Hazard + ·•` ` ·  .* `· .+·.* ·. ·
# .` *·` .·  · ·. `    https://adventofcode.com/2015/day/6  ·  + ·  .•·`. · ` · 
# ·. · ·` *·` ·  .·*`  .  ·        .·+   ·*.   ·+·. `·  .·+*.`  ·  ·  `·*    ·`.


import numpy as np


def parse(s: str):
    """Parses input into a list of intructions in the form
        (instruction, coords1, coords2)."""

    res = []

    for line in s.split("\n"):
        parts = line.split(" ")
        coords = [parts[-3], parts[-1]]

        coords1, coords2 = [tuple(int(elem) for elem in s.split(",")) for s in coords]
        instruction = " ".join(parts[:-3])

        res.append((instruction, coords1, coords2))

    return res


def run_instructions(instructions, shape=(1000, 1000), cumulative=False):
    """Updates the array of Christmas lights when lights are turned on/off or toggled between on/off.
    Toggling is implemented as incrementing by 1 - we can just take the value mod 2 later on.
    If cumulative is True then lights don't turn on/off in a binary fashion, but instead the number
    of on/off signals accumulate"""

    lights = np.zeros(shape=shape, dtype=int)

    for instruction, coords1, coords2 in instructions:
        (i1, j1), (i2, j2) = coords1, coords2

        if instruction == "toggle":
            if cumulative:
                lights[i1:i2 + 1, j1:j2 + 1] += 2
            else:
                lights[i1:i2 + 1, j1:j2 + 1] = 1 - lights[i1:i2 + 1, j1:j2 + 1]
            #
        elif instruction == "turn on":
            lights[i1:i2 + 1, j1:j2 + 1] += 1
        elif instruction == "turn off":
            lights[i1:i2 + 1, j1:j2 + 1] -= 1
        else:
            raise ValueError('Invalid instruction')

        # on/off lights cap at 0 or 1. Lights with brightness settings can't go below brightness 0
        max_ = float("inf") if cumulative else 1
        lights[i1:i2 + 1, j1:j2 + 1] = np.clip(lights[i1:i2 + 1, j1:j2 + 1], 0, max_)

    return lights


def solve(data: str) -> tuple[int|str, int|str]:
    instructions = parse(data)

    lights = run_instructions(instructions)
    star1 = sum(v % 2 for v in lights.flat)

    print(f"Solution to part 1: {star1}")

    lights2 = run_instructions(instructions, cumulative=True)

    star2 = sum(lights2.flat)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 6
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()