from copy import deepcopy
import numpy as np


def parse(s):
    chunks = s.split("\n\n")
    arrs = []
    for chunk in chunks:
        tups = [[int(val) for val in line.strip().split(",")] for line in chunk.split("\n")[1:]]
        # Parse coords like [[x1, x2, ...], [y1, y2, ...], ... ]
        coords = [np.array(x) for x in zip(*tups)]
        arrs.append(coords)
    return arrs


with open("input19.txt") as f:
    raw = f.read()


def iterate_with_sign(iterable):
    """Generates pairs like +/- 1, elem for each elem in iterable."""
    for sign in (-1, 1):
        for elem in iterable:
            yield sign, elem


def pop_direction(arr, ind):
    """Takes a list and an index. Returns the element at that index, and a copy
    of the list with the element removed."""
    remaining = deepcopy(arr)
    direction = remaining.pop(ind)
    return direction, remaining


def generate_orientations(coords):
    """Takes a list of 3 lists, representing e.g. N x, y, and z-coordinates, in the format
    [[x1, x2, ...], [y1, ...], ...]
    Generates the representations of the coordinates in each of the 8 possible orientations."""
    for sign_up, ind_up in iterate_with_sign(range(len(coords))):
        print(sign_up, ind_up)
        up, planar = pop_direction(coords, ind_up)
        up *= sign_up
        for sign_forward, ind_forward in iterate_with_sign(list(range(len(planar)))[::-1]):
            forward, straight = pop_direction(planar, ind_forward)
            forward *= sign_forward
            right = -straight[0]
            reoriented = deepcopy([up, forward, right])
            V = np.array(list(zip(*reoriented)))
            yield V


coordslists = parse(raw)
coords = coordslists[0]

uuuuugh = parse("""--- scanner 0 ---
-1,-1,1
-2,-2,2
-3,-3,3
-2,-3,1
5,6,-4
8,0,7""")[0]

hmm = 0
for stuff in generate_orientations(uuuuugh):
    print(stuff)
    print("\n")