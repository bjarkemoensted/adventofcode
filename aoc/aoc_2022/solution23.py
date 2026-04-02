# * .`+ ·`• ·.   + `. ·*  `   ·· `         ·. `*      `.   +·*`*·    · .`+*· .·`
# `· +·   ` .*`   .·+    `·.· ` Unstable Diffusion `·     .`  ··` ·+ `  .·  ·*`.
# . ·` ·`. * ·.·  `·   https://adventofcode.com/2022/day/23  ·* .·  `. *· .* ·.`
# `.*·`. · ·   +. ·  ·`*·    .+  ·`• ·.` · *`  .  ` .**·  · `  .+ `·  .·  `·*.`·

from collections import Counter
from functools import cache

import numpy as np
from numpy.typing import NDArray

type coordtype = tuple[int, int]

def parse(s: str) -> set[coordtype]:
    lines = s.splitlines()
    rows = len(lines)

    d = set([])

    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == "#":
                x = j
                y = rows - 1 - i
                d.add((x, y))
            #
        #
    return d


def _make_array(coords: set[coordtype]) -> NDArray[np.str_]:
    xvals, yvals = zip(*coords)
    xmin = min(xvals)
    ymin = min(yvals)
    translated = [(x - xmin, y - ymin) for x, y in coords]

    cols = max(yt for _, yt in translated)
    rows = max(xt for xt, _ in translated)

    data = [["." for _ in range(cols+1)] for _ in range(rows+1)]
    M = np.array(data)

    for i, j in translated:
        M[i, j] = "#"

    # Mess around with M so it looks right (transform x, y-coordinates into i,j (matrix) coords)
    res = np.flip(M.T, 0)
    return res


def display(coords: set[coordtype]) -> None:
    """Prints the current state in a way similar to the problem formulation."""
    M = _make_array(coords)
    lines = []
    for row in M:
        lines.append("".join(row))

    s = "\n".join(lines)
    print(s)


@cache
def _get_vicinity(point_xy: coordtype) -> list[coordtype]:
    """Returns the 8 points around input point"""
    res = []
    x0, y0 = point_xy
    for delta_x in (-1, 0, 1):
        for delta_y in (-1, 0, 1):
            if delta_y == delta_x == 0:
                continue
            res.append((x0 + delta_x, y0 + delta_y))
        #
    return res


def direction_free(dir_: str, point_xy: coordtype, occupied: set[coordtype]) -> bool:
    """Returns whether an elf at input point can move in the specified direction (NSWE)"""
    vicinity = _get_vicinity(point_xy)

    x0, y0 = point_xy
    danger_zone = None
    if dir_ == "N":
        danger_zone = {(x, y) for x, y in vicinity if y == y0 + 1}
    elif dir_ == "S":
        danger_zone = {(x, y) for x, y in vicinity if y == y0 - 1}
    elif dir_ == "W":
        danger_zone = {(x, y) for x, y in vicinity if x == x0 - 1}
    elif dir_ == "E":
        danger_zone = {(x, y) for x, y in vicinity if x == x0 + 1}
    else:
        raise ValueError

    return not any(coord in occupied for coord in danger_zone)


@cache
def get_adjacent_points_in_direction(point_xy: coordtype, dir_: str) -> set[coordtype]:
    """Returns a set of the adjacent points in the specified direction"""
    vicinity = _get_vicinity(point_xy)

    x0, y0 = point_xy
    if dir_ == "N":
        return {(x, y) for x, y in vicinity if y == y0 + 1}
    elif dir_ == "S":
        return {(x, y) for x, y in vicinity if y == y0 - 1}
    elif dir_ == "W":
        return {(x, y) for x, y in vicinity if x == x0 - 1}
    elif dir_ == "E":
        return {(x, y) for x, y in vicinity if x == x0 + 1}
    else:
        raise ValueError


def shift_point(point_xy: coordtype, dir_: str) -> coordtype:
    """Shifts point one step in the specified direction (NSWE)"""
    x, y = point_xy
    if dir_ == "N":
        y += 1
    elif dir_ == "S":
        y -= 1
    elif dir_ == "W":
        x -= 1
    elif dir_ == "E":
        x += 1
    else:
        raise ValueError

    return x, y


def propose_directions(occupied: set[coordtype], directions: list[str]):
    """Makes each elf occupying a site propose a new site for them to move to"""

    point2prop = {}
    for point_xy in occupied:
        point2prop[point_xy] = point_xy
        vicinity = _get_vicinity(point_xy)
        if not any(nearby_point in occupied for nearby_point in vicinity):
            continue
        for dir_ in directions:

            candidate_points = get_adjacent_points_in_direction(point_xy, dir_)
            is_free = candidate_points.isdisjoint(occupied)

            if is_free:
                proposed_point = shift_point(point_xy, dir_)
                point2prop[point_xy] = proposed_point
                break
            #
        #
    return point2prop


def update_occupations(occupied: set[coordtype], proposed: dict[coordtype, coordtype]) -> set[coordtype]:
    """Takes current occupations (set of xy-coords) amd proposed new coordinates (dict mapping xy coords to new coords),
    and applies the update rules (moves the elves which are alone in proposing the new position)."""

    new_occupations = set([])
    site2n_proposals = Counter(proposed.values())
    conflicts = {site for site, n in site2n_proposals.items() if n > 1}
    for old in occupied:
        new = proposed[old]
        if new in conflicts:
            new_occupations.add(old)
        else:
            new_occupations.add(new)
        #

    return new_occupations


def _cycle_directions(directions: list[str]) -> list[str]:
    """Cycles the list of directions, e.g. NSWE -> SWEN"""
    res = [s for s in directions[1:]] + [directions[0]]
    return res


def simulate_elves(occupied: set[coordtype], n_rounds_max=-1) -> tuple[set[coordtype], int]:
    """Simulates the procedure used by the elves to spread out during their farming shenanigans."""

    stop_at = n_rounds_max if n_rounds_max != -1 else float("inf")
    directions = list("NSWE")
    round_number = 0

    while round_number < stop_at:
        round_number += 1
        proposed = propose_directions(occupied, directions)
        new_occupied = update_occupations(occupied, proposed)
        if new_occupied == occupied:
            break
        occupied = new_occupied
        directions = _cycle_directions(directions)

    return occupied, round_number



def solve(data: str) -> tuple[int|str, ...]:
    occupied_initial = parse(data)

    # Simulate the initial 10 rounds
    n_rounds = 10
    occupied_final, _ = simulate_elves(occupied_initial, n_rounds_max=n_rounds)
    M = _make_array(occupied_final)
    star1 = sum(site == "." for site in M.flat)
    print(f"Solution to part 1: {star1}")

    # Start over and keep going until their positions stabilize.
    _, star2 = simulate_elves(occupied_initial)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 23
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
