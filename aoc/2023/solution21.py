import numpy as np


def read_input():
    with open("input21.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    M = np.array([list(line) for line in s.split("\n")])
    return M


north = (-1, 0)
west = (0, -1)
east = (0, 1)
south = (1, 0)

all_dirs = (north, west, south, east)


def get_neighbors(M, coords):
    """Takes an array and an iterable of coordinates with format (i, j).
    Generates all neighboring coordinates that don't fall off the array. Might yield duplicates."""
    for crd in coords:
        for dir_ in all_dirs:
            neighbor_crd = tuple(x+delta for x, delta in zip(crd, dir_))
            if all(0 <= x < lim for x, lim in zip(neighbor_crd, M.shape)) and M[neighbor_crd] != "#":
                yield neighbor_crd
            else:
                pass


def get_starting_coord(M):
    """Determines the starting coordinate"""
    a, b = np.where(M == "S")
    assert all(len(crd) == 1 for crd in (a, b))
    return a[0], b[0]


def get_positions_possible_after_steps(M, n_steps):
    """Determines the locations (garden plots) the gardener can reach after exactly n steps."""
    M = M.copy()
    start = get_starting_coord(M)
    coords = [start]

    for _ in range(n_steps):
        newcoords = set(get_neighbors(M, coords))
        coords = list(newcoords)

    return coords


def determine_parities(M):
    """Detects a recurring pattern in the coordinates reachable on M after an equal/odd number of steps.
    Returns a tuple of the coordinates reachable after equal/odd steps after a cycle has begun."""

    parities = [None, None]
    start = get_starting_coord(M)
    coords = [start]
    seen = set([])

    n_steps = 0
    while any(elem is None for elem in parities):
        newcoords = set(get_neighbors(M, coords))
        coords = list(newcoords)
        n_steps += 1
        key = tuple(coords)
        if key in seen:
            parind = n_steps % len(parities)
            parities[parind] = coords
        else:
            seen.add(key)

    return parities


def get_n_positions_on_infitinite_grid(M, n_steps):
    """Computes the number of positions reachable after exactly n steps on an infinite grid of tiles like M.
    Each tile forms a rhombus at the center, with some additional garden plots in the corners.
    This method works by figuring out how many plots are reachable on a tile given the parity (equal/odd) of the number
    of steps. For tiles at the edge of the reachable rhombus, only some subsets of the tile (the center and/or a subset
    of the corners) is reachable, so those are added at the end."""

    rows, cols = M.shape
    assert rows == cols
    center = rows // 2

    # Determine the reachable tiles by parity for full tiles, and center/corner segments
    parity_coords = determine_parities(M)
    parity_full = [len(crds) for crds in parity_coords]  # plots reachable if tile is full covered

    # Determine also the number of plots reachable in the center and outer edges of the tile
    parity_middle = [sum(sum(abs(x - center) for x in (i, j)) <= center for i, j in crds) for crds in parity_coords]
    parity_outer = [full - middle for full, middle in zip(parity_full, parity_middle)]

    # Start by computing the reachable plots in all the tiles fully covered by rhombus the gardener can reach
    res = 0
    n_cells_length = n_steps // rows
    steps_remaining = n_steps
    i = 0
    # The center tile has odd parity of n_steps is odd, and vice versa
    parind_center = n_steps % len(parity_full)

    while steps_remaining > rows:
        cell_area = parity_full[(i+parind_center) % len(parity_full)]
        # Just grab the fully covered til in one corner, e.g. north-east of starting point, and multiply by 4
        n_cells = 1 if i == 0 else 4*i
        delta = n_cells*cell_area
        res += delta
        i += 1
        steps_remaining -= rows

    assert i == n_cells_length

    # Along the edge of the rhombus are segments only partially covered. Determine their parity
    parind_point = (n_cells_length + parind_center) % len(parity_full)
    parind_big = parind_point
    parind_small = (parind_big + 1) % len(parity_full)

    # add the n-1 'big' remainders
    bigrem = (n_cells_length - 1)*(4*parity_full[parind_big] - parity_outer[parind_big])
    res += bigrem

    # add the n 'small' remainders
    smallrem = n_cells_length*parity_outer[parind_small]
    res += smallrem

    # add the 4 'point' remainders
    pointrem = 4*parity_middle[parind_point] + 2*parity_outer[parind_point]
    res += pointrem

    return res


def main():
    raw = read_input()
    M = parse(raw)

    n_steps = 64
    locs = get_positions_possible_after_steps(M, n_steps=n_steps)
    star1 = len(locs)
    print(f"After {n_steps} steps, the gardener can reach {star1} distinct locations.")

    n_steps2 = 26501365
    star2 = get_n_positions_on_infitinite_grid(M, n_steps2)
    print(f"After {n_steps2} steps on an infinite grid, the gardener can reach {star2} locations.")


if __name__ == '__main__':
    main()
