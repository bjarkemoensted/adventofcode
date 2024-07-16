import numpy as np


def read_input():
    with open("input13.txt") as f:
        puzzle_input = f.read()

    return puzzle_input


def parse(s):
    maps = []
    for part in s.split("\n\n"):
        M = np.array([list(line) for line in part.split("\n")])
        maps.append(M)
    return maps


def slice(M, ind, axis):
    """Slices an array (2D) at row/column ind along the specified axis.
    For example, ind=4, axis=0 cuts the array after the 4th row.
    Returns the two parts into which the array is sliced in this way.
    The part below/to the right of the cut is flipped and rows/cols are truncated to
    facilitate easy identification of symmetries."""

    nrows, ncols = M.shape
    if axis == 0:
        cut = min(ind, nrows - ind)
        a = M[:ind, :][-cut:, :]
        b = np.flip(M[ind:, :][:cut, :], axis=axis)
    elif axis == 1:
        cut = min(ind, ncols - ind)
        a = M[:, :ind][:, -cut:]
        b = np.flip(M[:, ind:][:, :cut], axis=axis)
    else:
        raise ValueError("hmm")
    if a.shape != b.shape:
        raise ValueError(f"Wrong shapes: {a.shape} and {b.shape}. M has shape {M.shape}. Used ind {ind}, axis {axis}.")
    return a, b


def find_symmetry_line(M, n_mismatches_needed=0):
    """Finds the row/col index and axis along which there's a symmetry in the input array.
    n_mismatches_needed can be set to 0 for exact symmetry, or to 1 for symmetry except for a single smudge."""
    axes = (0, 1)
    for limit, axis in zip(M.shape, axes):
        for i in range(limit):
            a, b = slice(M, ind=i, axis=axis)
            diff = (a != b).flat
            if len(diff) > 0 and sum(diff) == n_mismatches_needed:
                return i, axis


def summarize_notes(maps, n_mismatches_needed=0):
    res = 0
    for M in maps:
        i, axis = find_symmetry_line(M, n_mismatches_needed=n_mismatches_needed)
        score = i
        if axis == 0:
            score *= 100
        res += score

    return res


def main():
    raw = read_input()
    maps = parse(raw)

    for M in maps:
        find_symmetry_line(M)

    star1 = summarize_notes(maps)
    print(f"The notes summarize to: {star1}.")

    star2 = summarize_notes(maps, n_mismatches_needed=1)
    print(f"The notes summarize to: {star2}.")


if __name__ == '__main__':
    main()
