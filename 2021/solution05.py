from collections import defaultdict
import copy
import numpy as np


def parse_point(s):
    return np.array([int(x) for x in s.strip().split(",")])


def parse_line(s):
    point_strings = s.strip().split(" -> ")
    return tuple(parse_point(ps) for ps in point_strings)


# Read in data on lines of vulcanic activity, format ((x1, y1), (x2, y2)) - end points of lines
with open("input05.txt") as f:
    data = [parse_line(line) for line in f]


def line_is_vertical_or_horizontal(line):
    (x1, y1), (x2, y2) = line
    return x1 == x2 or y1 == y2


# Include only horizontal and vertical lines
valid_lines = [line for line in data if line_is_vertical_or_horizontal(line)]


def get_points_on_line(line):
    """Generator for all points lying on a given line"""
    line = copy.deepcopy(line)
    p1, p2 = line
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    # Ugly hack to get direction to be [+-1, +-1] for diagnonals
    direction = np.round((vec * 1 / norm)).astype(int)
    running = p1
    n_steps = max(map(abs, vec))
    for i in range(n_steps + 1):
        yield running
        running = running + direction


def count_active_spots(lines):
    """Counts number of spots with active vulcanoes"""
    lines = copy.deepcopy(lines)
    counts = defaultdict(lambda: 0)
    for line in lines:
        for point in get_points_on_line(line):
            key = tuple(point)
            counts[key] += 1
        #
    return counts


spot_counts = count_active_spots(valid_lines)
n_overlaps = sum(v >= 2 for v in spot_counts.values())
print(f"Solution for star 1: {n_overlaps}")


def line_is_diagonal(line):
    p1, p2 = line
    delta_x, delta_y = p2 - p1
    res = abs(delta_x) == abs(delta_y)
    return res


# Redo analysis including diagonal lines
new_lines = [line for line in data if line_is_vertical_or_horizontal(line) or line_is_diagonal(line)]
counts_with_diagnonals = count_active_spots(new_lines)
n_overlaps2 = sum(v >= 2 for v in counts_with_diagnonals.values())

print(f"Solution to star 2: {n_overlaps2}.")
