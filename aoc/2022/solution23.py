from collections import Counter
from copy import deepcopy
import numpy as np


def read_input():
    with open("input23.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    lines = s.split("\n")
    rows = len(lines)

    d = set([])

    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == "#":
                x = j
                y = rows - 1 - i
                d.add((x, y))

    return d


def _make_array(coords):
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


def display(coords):
    """Prints the current state in a way similar to the problem formulation."""
    M = _make_array(coords)
    lines = []
    for row in M:
        lines.append("".join(row))

    s = "\n".join(lines)
    print(s)


def _get_vicinity(point_xy):
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


def direction_free(dir_, point_xy, occupied):
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


def shift_point(point_xy, dir_):
    """Shofts point one step in the specified direction (NSWE)"""
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


def propose_directions(occupied, directions):
    """Makes each elf occupying a site propose a new site for them to move to"""
    point2prop = {}
    for point_xy in occupied:
        point2prop[point_xy] = point_xy
        vicinity = _get_vicinity(point_xy)
        if not any(nearby_point in occupied for nearby_point in vicinity):
            continue
        for dir_ in directions:
            if direction_free(dir_, point_xy, occupied):
                proposed_point = shift_point(point_xy, dir_)
                point2prop[point_xy] = proposed_point
                break
            #
        #
    return point2prop


def update_occupations(occupied, proposed):
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


def _cycle_directions(directions):
    """Cycles the list of directions, e.g. NSWE -> SWEN"""
    res = [s for s in directions[1:]] + [directions[0]]
    return res


def simulate_elves(occupied, n_rounds_max=None):
    """Simulates the procedure used by the elves to spread out during their farming shenanigans."""
    if n_rounds_max is None:
        n_rounds_max = float("inf")
    directions = list("NSWE")
    occupied = deepcopy(occupied)
    round_number = 0

    while round_number < n_rounds_max:
        round_number += 1
        proposed = propose_directions(occupied, directions)
        new_occupied = update_occupations(occupied, proposed)
        if new_occupied == occupied:
            break
        occupied = new_occupied
        directions = _cycle_directions(directions)

    return occupied, round_number


def main():
    raw = read_input()
    occupied_initial = parse(raw)

    # Simulate the initial 10 rounds
    n_rounds = 10
    occupied_final, _ = simulate_elves(occupied_initial, n_rounds_max=n_rounds)
    M = _make_array(occupied_final)
    free_area = sum(site == "." for site in M.flat)
    print(f"Free area after {n_rounds} rounds is {free_area}.")

    # Start over and keep going until their positions stabilize.
    _, last_round = simulate_elves(occupied_initial)
    print(f"The elves stabilize after {last_round} rounds.")


if __name__ == '__main__':
    main()
