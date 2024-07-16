import numpy as np

# Read in data
with open("input18.txt") as f:
    puzzle_input = f.read()


example_input = \
""".#.#.#
...##.
#....#
..#...
#.#..#
####.."""


def parse(s):
    """Parses input into an array of ones and zeroes indicating lights on/off."""
    lines = [list(line) for line in s.split("\n")]
    res = np.zeros(shape=(len(lines), len(lines[0])), dtype=int)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == "#":
                res[i, j] = 1

    return res


def count_neighbors_on(arr, i, j):
    rows, cols = arr.shape
    up = max(0, i-1)
    down = min(rows, i+2)
    left = max(0, j-1)
    right = min(cols, j+2)

    subarr = arr[up:down, left:right]
    n_on_in_area = sum(subarr.flat)
    n_neighbors_on = n_on_in_area - arr[i, j]

    return n_neighbors_on


def display_lights(matrix):
    d = {1: "#", 0: "."}
    lines = []
    for row in matrix:
        line = [d[val] for val in row]
        lines.append("".join(line))
    out = "\n".join(lines)
    print(out)


def coords(arr):
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            yield i, j


def turn_on_corners(arr):
    rows, cols = arr.shape
    for i, j in ((0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)):
        arr[i, j] = 1


def iterate(arr, n, corners_stay_on=False):
    res = arr.copy()
    for _ in range(n):
        new_setting = np.zeros(shape=res.shape, dtype=int)
        for i, j in coords(res):
            n = count_neighbors_on(res, i, j)
            new_val = None
            if res[i, j] == 1:
                new_val = 1 if n in (2, 3) else 0
            elif res[i, j] == 0:
                new_val = 1 if n == 3 else 0
            new_setting[i, j] = new_val
        if corners_stay_on:
            turn_on_corners(new_setting)
        res = new_setting
    return res


m = parse(puzzle_input)
m = iterate(m, n=100)
print(f"After 100 iterations, {sum(m.flat)} lights are on.")

m2 = parse(puzzle_input)
turn_on_corners(m2)
m2 = iterate(m2, n=100, corners_stay_on=True)
print(f"When forcing corner lights on, {sum(m2.flat)} lights are on.")
