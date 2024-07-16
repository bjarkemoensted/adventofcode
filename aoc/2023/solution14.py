import numpy as np


def read_input():
    with open("input14.txt") as f:
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


def _display(M):
    """Represents the platform as a string"""
    lines = []
    for i in range(M.shape[0]):
        lines.append("".join(M[i, :]))
    s = "\n".join(lines)
    return s


def _iterate(M, dir_):
    """Iterates over the sites on the platform in a way that makes sense given the direction in which things slide.
    If things slide e.g. north, we start from the north, so the rocks furthest north move first, making room for
    the ones further south."""

    rows, cols = M.shape
    if dir_ not in all_dirs:
        raise ValueError

    # Switch the direction if we're iterating from the south or east
    rowiter = range(rows-1, -1, -1) if dir_ == south else range(rows)
    coliter = range(cols-1, -1, -1) if dir_ == east else range(cols)
    for i in rowiter:
        for j in coliter:
            yield i, j


def get_neighbor(M, crd, dir_):
    """Gets the site one step in the input direction from the input coordinates.
    If the steps falls off the edge of the platform, returns None."""
    crds = tuple(x+delta for x, delta in zip(crd, dir_))
    if all(0 <= x < lim for x, lim in zip(crds, M.shape)):
        return crds
    else:
        return None


def slide(M, dir_):
    """Takes an array M representing the platform. Repeatedly slides rocks in the input direction,
    until nothing moves."""

    updated = np.copy(M)
    for crd in _iterate(updated, dir_):
        # Check that there's a site on the platform one step in the direction
        ncrd = get_neighbor(updated, crd, dir_)
        if ncrd is None:
            continue

        # Check if there's a rock at current site, and free space at neighboring site
        char = updated[crd]
        nchar = updated[ncrd]
        rock = char == "O"
        roll = nchar == "."
        if rock and roll:  # heh
            updated[ncrd] = "O"
            updated[crd] = "."
        #

    if np.all(M == updated):
        return updated  # If nothing moved, the rocks are done sliding
    else:
        return slide(updated, dir_)


def compute_load(M):
    res = 0
    nrows, _ = M.shape
    for i in range(nrows):
        dist = nrows - i
        n_rocks = sum(c == "O" for c in M[i, :])
        res += dist*n_rocks

    return res


def cycle(M, n=1):
    # Keep track of which states we've encountered
    key2ind = {}
    states = []
    # Identify if a loop starts
    loop_start = None
    loop_end = None

    for i in range(n):
        states.append(M)
        key = _display(M)
        seen_before = key in key2ind
        # If a loop is identified, there's no need to compute omre cycles
        if seen_before:
            loop_start = key2ind[key]
            loop_end = i
            break

        key2ind[key] = i
        # Cycle to the next state
        for dir_ in all_dirs:
            M = slide(M, dir_)
        #

    # If we didn't encounter a loop, all n cycles were done, so just return the final state.
    if loop_start is None:
        return M

    # If a loop was identified, ignore all trips around the loop
    steps_remaining_loop_enter = n - loop_start
    loop_size = loop_end - loop_start
    n_effective_steps = steps_remaining_loop_enter % loop_size
    ind_final = loop_start + n_effective_steps
    M_final = states[ind_final]

    return M_final


def main():
    raw = read_input()
    M = parse(raw)

    M2 = slide(M, dir_=north)
    star1 = compute_load(M2)
    print(f"The total load after sliding the rocks north is {star1}.")

    n = 10**9
    M3 = cycle(M, n=n)

    star2 = compute_load(M3)
    print(f"Total load after {n} cycles: {star2}.")


if __name__ == '__main__':
    main()
