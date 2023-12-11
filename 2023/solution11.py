import numpy as np


def read_input():
    with open("input11.txt") as f:
        puzzle_input = f.read()

    return puzzle_input


def parse(s):
    lines = s.split("\n")
    M = np.array([list(line) for line in lines])

    return M


def get_empty_rows_and_cols(M):
    """Returns tuples of the indices of rows and columns containing empty space.
    Format is e.g. ([2, 4, 6], [1, 13])"""
    res = np.copy(M)
    rows, cols = res.shape
    empty_rows = [i for i in range(rows) if all(char == "." for char in M[i])]
    empty_cols = [j for j in range(cols) if all(char == "." for char in M[:, j])]

    return empty_rows, empty_cols


def get_galaxy_coords(M):
    """Returns the coordinates (i, j) of galaxies"""
    res = []
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            if M[i, j] == "#":
                res.append((i, j))
            #
        #

    return res


def expand_coord(coord, empty_space, expansion_dist):
    """Expands the input coordinates. each row and column containing empty space is expanded by a factor of
    expansion_dist."""
    res = []
    for x, emp in zip(coord, empty_space):
        # Every row/col with empty space adds expansion_dist - 1 to the resulting coordinate element
        res.append(x + (expansion_dist - 1)*sum(x > val for val in emp))

    return tuple(res)


def manhatten_dist(a, b):
    res = sum(abs(x1 - x2) for x1, x2 in zip(a, b))
    return res


def sum_of_pairwise_dists(coords, empty_space, expansion_dist):
    res = 0

    # Take into account the expansion of the empty rows/cols before computing the distances
    coords = [expand_coord(coord, empty_space, expansion_dist) for coord in coords]
    for i, a in enumerate(coords):
        for j, b in enumerate(coords[i + 1:]):
            res += manhatten_dist(a, b)
        #
    return res


def main():
    raw = read_input()
    M = parse(raw)

    coords = get_galaxy_coords(M)
    empty_space = get_empty_rows_and_cols(M)

    star1 = sum_of_pairwise_dists(coords, empty_space, expansion_dist=2)
    print(f"Sum of pair distances: {star1}.")

    star2 = sum_of_pairwise_dists(coords, empty_space, expansion_dist=10**6)
    print(f"Sum of pair distances with new expansion: {star2}.")


if __name__ == '__main__':
    main()
