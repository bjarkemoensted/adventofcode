import copy
import numpy as np
import re


def read_input():
    with open("input14.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def get_points_on_line(point_a, point_b):
    """Returns all points on an array on a line from a to b"""
    point_a = np.array(point_a)
    point_b = np.array(point_b)

    direction = point_b - point_a
    step = np.clip(direction, -1, 1)

    running = point_a
    points_on_line = [tuple(point_a)]
    while not np.all(running == point_b):
        running += step
        points_on_line.append(tuple(running))

    return points_on_line


def parse(s, add_floor=False):
    """Parses input data into a numpy array"""

    # Find all the coordinates in input
    coords = re.findall(r"(\d+),(\d+)", s)

    # Determine the x and y range required to draw the 'map'.
    limits = []
    for arr in zip(*coords):
        limits.append(max(map(int, arr)))
    xlim, ylim = limits
    # Add some space for the floor (puzzle part 2)
    ylim += 2
    x_required = 500 + ylim
    if xlim < x_required:
        xlim = x_required

    # Use '.' for empty space and '#' for walls
    M = np.array([["." for _ in range(xlim+1)] for _ in range(ylim+1)])

    for line in s.split("\n"):
        # Connect all the points specified in the line
        parts = [tuple(int(val) for val in substring.split(",")) for substring in line.split(" -> ")]
        for i in range(len(parts)-1):
            a = parts[i]
            b = parts[i+1]
            for x, y in get_points_on_line(a, b):
                M[y, x] = "#"
            #
        #

    # Create a floor for part 2
    if add_floor:
        for x, y in get_points_on_line((0, ylim), (xlim, ylim)):
            M[y, x] = "#"

    return M


def trickle(M, origin_yx):
    """Make a grain of sand trickle from origin point down through the map."""

    running = np.array(origin_yx)
    ylim, xlim = M.shape
    at_rest = False
    occupied = {'#', 'O'}  # sand stops if it hits a wall, or another grain of sand

    # It tries to move down, or, if blocked, left or right.
    step_directions = [np.array(tup) for tup in ((1, 0), (1, -1), (1, 1))]
    while not at_rest:
        took_step = False
        for step in step_directions:
            # Try to update position
            next_point = running + step
            ii, jj = next_point

            # Complain if we go out of bounds
            out_of_bounds = ii >= ylim or jj >= xlim or any(val < 0 for val in next_point)
            if out_of_bounds:
                raise IndexError

            # Update position if possible, then restart update procedure
            if M[ii, jj] not in occupied:
                running = next_point
                took_step = True
                break
            #
        # Done if the sand grain cannot move anymore
        at_rest = not took_step

    # Up date the map with the grain of sand that has now come to rest
    y, x = running
    M[y, x] = "O"


def fill_with_sand(M, origin_yx=None):
    """Fills map with sand until sand starts falling off the map."""

    M = copy.deepcopy(M)
    if origin_yx is None:
        origin_yx = (0, 500)

    # Just keep adding more sand until we're out of space
    space = True
    while space:
        try:
            trickle(M, origin_yx=origin_yx)
        except IndexError:
            space = False
        #
    return M


def fill_until_blocked(M, origin_yx=None):
    """Keeps filling the map with sand until the sand fills up to its entry point."""

    M = copy.deepcopy(M)
    if origin_yx is None:
        origin_yx = (0, 500)

    i0, j0 = origin_yx
    n_units = 0
    while M[i0, j0] != "O":
        try:
            trickle(M, origin_yx=origin_yx)
            n_units += 1
            print(n_units, end="\r")
        except IndexError:
            raise
        #
    return M


def main():
    raw = read_input()
    M = parse(raw)

    M_full = fill_with_sand(M)
    print(f"After the cave fills, there is {sum(M_full.flat == 'O')} units of sand.")

    M_floor = parse(raw, add_floor=True)
    M_full2 = fill_until_blocked(M_floor)
    print(f"After the cave fills completely, there is {sum(M_full2.flat == 'O')} units of sand.")


if __name__ == '__main__':
    main()
