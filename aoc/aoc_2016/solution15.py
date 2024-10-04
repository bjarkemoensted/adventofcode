import math
import re

_test = """Disc #1 has 5 positions; at time=0, it is at position 4.
Disc #2 has 2 positions; at time=0, it is at position 1."""


def read_input():
    with open("input15.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


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


def solve(discs, maxiter=None):
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


def main():
    raw = read_input()
    discs = parse(raw)

    star1 = solve(discs)
    print(f"The capsule makes it through the machine if dropped after (multiples of) {star1} seconds.")

    additional_disc = (11, 0)
    discs.append(additional_disc)

    star2 = solve(discs)
    print(f"With the extra disc (multiples of) {star2} seconds are required.")


if __name__ == '__main__':
    main()
