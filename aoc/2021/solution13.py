import copy

import numpy as np

with open("input13.txt") as f:
    chunks = f.read().split("\n\n")

# Read in data to represent dots on the transparent paper
xinds, yinds = zip(*[map(int, line.strip().split(",")) for line in chunks[0].split("\n")])

M = np.zeros(shape=(max(yinds) + 1, max(xinds) + 1), dtype=int)
for i, j in zip(yinds, xinds):
    M[i, j] = 1

# Read in folding instructions, e.g. ('x', 42) for folding at the x=42 line
folding_instructions = []
for line in chunks[1].split("\n"):
    direction, val = line.split("fold along ")[1].split("=")
    val = int(val)
    folding_instructions.append((direction, val))


def fold(m, axis, coord, verbose=False):
    """Returns a folded paper. For example, pass m=some matrix, axis='y', coord + 42.
    This will return the input matrix folded around y=42. The lower part is flipped and
    added to the upper part, like when folding a paper."""
    if verbose:
        print(f"Folding matrix with shape {m.shape}, around {axis}={coord}")

    # Copy data so we don't mess anything up
    m = copy.deepcopy(m)
    # If we're folding around x=something, partition into a left and right half
    if axis == 'x':
        ma, mb = m[:, :coord], m[:, coord+1:]
        flipaxis = 1
    # Similarly, partition into upper and lower when flipping around y=...
    elif axis == "y":
        ma, mb = m[:coord, :], m[coord+1:, :]
        flipaxis = 0
    else:
        raise ValueError

    # The 'halves' don't always line up exactly, so zero-pad the latter half so the shapes match.
    b2 = np.zeros(shape=ma.shape)
    b2[:mb.shape[0], :mb.shape[1]] = mb
    if verbose:
        print(f"Shapes: a={ma.shape}, b={mb.shape}. b2 = {b2.shape}")
    mb = b2

    if ma.shape != mb.shape:
        raise ValueError(f"Matrix a shape ({ma.shape}) != matrix b shape ({mb.shape})")

    # Flip, add and return
    ma = copy.deepcopy(ma)
    mb = np.flip(copy.deepcopy(mb), axis=flipaxis)
    res = ma + mb
    return res


n_nonzero_after_first_fold = 0
# Carry out all the folding instructions
for i, (dir_, coord) in enumerate(folding_instructions):
    M = fold(M, dir_, coord)
    if i == 0:
        n_nonzero_after_first_fold = sum(v >= 1 for v in M.flat)
        print(n_nonzero_after_first_fold)

print(f"Solution to star 1: {n_nonzero_after_first_fold}.")

# Make results more readable by converting zeroes to '.' and nonzeroes to '#'
M_display = np.zeros(shape=M.shape, dtype=str)
nrows, ncols = M.shape
for i in range(nrows):
    for j in range(ncols):
        M_display[i, j] = "#" if M[i, j] > 0 else "."

# The riddles states there are 8 letters in the solution. Print each in turn.
dispsize = 8
dispwidth = M_display.shape[1] // dispsize
for i in range(dispsize):
    disp = M_display[:, i*dispwidth:(i+1)*dispwidth]
    print(disp)