#  +  ⸳.•ꞏ * `  ⸳.ꞏ    .* ꞏ*  +ꞏ⸳ `*.  *.   .`⸳ꞏ  •*.`      .   ⸳+ꞏ `⸳ ` ꞏ * ⸳⸳ꞏ
#  `ꞏ⸳  •  ⸳ +.*  ` ꞏ. ⸳ ꞏ• .• Timing is Everything ꞏ .*     +     .* +ꞏ ⸳* ` ꞏ.
# .   ꞏ•  . +    .⸳+*  https://adventofcode.com/2016/day/15  `     ⸳.* ꞏ *ꞏ⸳ ` •
#  ꞏ*` ⸳ * `ꞏ ⸳`.• ꞏ • *⸳  ꞏ⸳  * ` ꞏ• ꞏ. ⸳⸳.    ⸳` ꞏ•*`.  *  ꞏ*⸳ꞏ  + .`*. .  * ⸳


import math
import re


def parse(s):
    res = []

    for line in s.split("\n"):
        pattern = r"Disc #(\d*) has (\d*) positions; at time=0, it is at position (\d*)."
        m = re.match(pattern, line)
        res.append(tuple(map(int, m.groups()[1:])))

    return res


def trace(time_start: int, discs: list):
    slot_position = 0
    """Determines how many discs a capsule released at the specified time will fall through"""
    time = time_start
    n_discs_passed = 0
    for n_positions, initial_position in discs:
        time += 1
        # Compute disc position at the updated time
        pos = (initial_position + time) % n_positions
        if pos == slot_position:
            n_discs_passed += 1
        else:
            break
        #

    return n_discs_passed


def determine_wait_seconds(discs, maxiter=None):
    """Determines the number of seconds to wait before a dropped capsule will make it through the machines.
    Initially, drops are attempted after 0, 1, 2, ... seconds, and the number of discs succesfully passed is computed.
    The increment is continually updated to the least common multiple of the periods of all discs passed in this manner.
    Similar approach to the Chinese Remainder Theorem, I think."""

    if maxiter is None:
        maxiter = float("inf")

    inc = 1
    time = 0
    n_its = 0
    while n_its < maxiter:
        n_discs_passed = trace(time_start=time, discs=discs)
        if n_discs_passed == len(discs):
            return time
        else:
            periods = [n_positions for n_positions, _ in discs[:n_discs_passed]]
            inc = math.lcm(*periods)
            time += inc
        n_its += 1


def solve(data: str):
    discs = parse(data)

    star1 = determine_wait_seconds(discs)
    print(f"The capsule makes it through the machine if dropped after (multiples of) {star1} seconds.")

    additional_disc = (11, 0)
    discs.append(additional_disc)

    star2 = determine_wait_seconds(discs)
    print(f"With the extra disc (multiples of) {star2} seconds are required.")

    return star1, star2


def main():
    year, day = 2016, 15
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
