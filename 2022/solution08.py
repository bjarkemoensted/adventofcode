import numpy as np


def read_input():
    with open("input08.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = np.array([[int(char) for char in line] for line in s.split("\n")])
    return res


def directions():
    d = dict(
        down=np.array([1, 0]),
        up=np.array([-1, 0]),
        right=np.array([0, 1]),
        left=np.array([0, -1])
    )
    return d


def iterate_direction(M, i, j, direction, skip_initial=False):
    """Iterates in a specified direction from starting coordinates in a matrix.
    If skip_initial is set, will not yield the first point.
    For instance, passing i, j = 1, 2, this will skip (1,2) and only yield
    e.g. (1, 3), (1, 4) and so on (assuming direction=[0, 1])."""

    coords = np.array([i, j])

    first = True
    while all(0 <= ind < edge for ind, edge in zip(coords, M.shape)):
        i, j = coords
        if not first or not skip_initial:
            yield i, j
        first = False
        coords += direction


def find_visible_trees(M):
    """Finds all trees visible from outside the forest."""

    d = directions()

    # Dict containing the coordinates for all points at the edge of the forest
    rows, cols = M.shape
    edges = dict(
        left=((i, 0) for i in range(rows)),
        right=((i, cols - 1) for i in range(rows)),
        upper=((0, j) for j in range(cols)),
        lower=((rows - 1, j) for j in range(cols))
    )

    # Map each edge to the direction we should scan (e.g. looking from the left, we should scan right across trees).
    edge2scan_direction = {
        "left": "right",
        "right": "left",
        "upper": "down",
        "lower": "up"
    }

    trees = set([])
    forest_peak = max(M.flat)  # Tallest tree in the forest. Stop scanning if we encounter this height.
    for edge_key, scandir_key in edge2scan_direction.items():
        # Get the points on one edge and the direction to scan from there
        edge_points = edges[edge_key]
        direction = d[scandir_key]
        for i, j in edge_points:
            tallest_so_far = -1
            for ii, jj in iterate_direction(M, i, j, direction):
                height = M[ii, jj]

                is_visible = height > tallest_so_far
                if is_visible:
                    trees.add((ii, jj))

                # Keep track of the tallest tree seen so far for line of sight computation
                if height > tallest_so_far:
                    tallest_so_far = height

                # Stop scanning if we encounter a tree that'll block any further view.
                if height >= forest_peak:
                    break

    return trees


def determine_scenic_score(M, i, j):
    """Determine the 'scenic score' at position i, j in the forest.
    Scan across trees in all four direction until line of sight is broken."""

    res = 1
    stopat = M[i, j]
    for direction in directions().values():
        n_trees_in_direction = 0
        for ii, jj in iterate_direction(M, i, j, direction, skip_initial=True):
            height = M[ii, jj]
            n_trees_in_direction += 1
            if height >= stopat:
                break
            #
        res *= n_trees_in_direction

    return res


def determine_max_scenic_score(M):
    """Find the maximum scenic score iterating over all cells."""
    best = 0
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            score = determine_scenic_score(M, i, j)
            best = max(best, score)
        #

    return best


def main():
    raw = read_input()
    M = parse(raw)

    visible_trees = find_visible_trees(M)
    print(f"There are {len(set(visible_trees))} visible trees.")

    best_scenic_score = determine_max_scenic_score(M)
    print(f"The maximum scenic score is {best_scenic_score}.")


if __name__ == '__main__':
    main()
