import numpy as np


def read_input():
    with open("input16.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = np.array([list(line) for line in s.split("\n")])
    return res


up = (-1, 0)
right = (0, 1)
down = (1, 0)
left = (0, -1)


def update_direction(crd, dir_, map_):
    """Returns the direction(s) of light beam(s) after a beam has visited input coordinate."""
    char = map_[crd]
    if char == ".":
        return (dir_,)
    elif char == "|":
        if dir_ in (up, down):
            return (dir_,)
        else:
            return (up, down)
    elif char == "-":
        if dir_ in (left, right):
            return (dir_,)
        else:
            return (right, left)
    elif char == r"/":
        return ({right: up, up: right, left: down, down: left}[dir_],)
    elif char == "\\":
        return ({right: down, down: right, left: up, up: left}[dir_],)
    else:
        raise ValueError(f"Can't update with char: {char}")


def step(crd, dir_, map_):
    """Updates a coordinate given a direction. Returns None if taking the step falls off the map"""
    newcrd = tuple(a + b for a, b in zip(crd, dir_))
    if not all (0 <= x < lim for x, lim in zip(newcrd, map_.shape)):
        return None
    return newcrd


def shine(map_, initial_wavefront=None):
    """Computes the final number of energized tiles given an initial wavefront (coord + direction tuple)."""

    if initial_wavefront is None:
        initial_wavefront = ((0, 0), (0, 1))

    energized = set([])  # Keep track of which tiles are energized
    wavefronts = {initial_wavefront}  # Keep track of the wavefronts we have to update
    wavefront_history = set({})  # Keep track of wavefronts we've seen before

    # Keep updating as long as we're encountering previously unseen wavefronts
    while wavefronts - wavefront_history:
        new_wavefronts = set([])

        for wavefront in wavefronts:
            wavefront_history.add(wavefront)
            crd, dir_ = wavefront
            energized.add(crd)
            dirs = update_direction(crd, dir_, map_)
            for newdir in dirs:
                newcrd = step(crd, newdir, map_)
                if newcrd is not None:
                    new_wavefront = (newcrd, newdir)
                    if new_wavefront not in wavefront_history:
                        new_wavefronts.add(new_wavefront)
            #
        wavefronts = new_wavefronts

    n_energized = len(energized)
    return n_energized


def find_highest_energization(map_):
    """Just try all sites along the edge of the map with all light beams going inwards."""

    rows, cols = map_.shape
    # Define all possible initial wavefronts whining inwards from the edges
    initial_wavefronts = []
    for i in range(rows):
        initial_wavefronts.append(((i, 0), right))  # left shining right
        initial_wavefronts.append(((i, cols - 1), left))  # right shining left
    for j in range(cols):
        initial_wavefronts.append(((0, j), down))  # up shining down
        initial_wavefronts.append(((rows - 1, j), up))  # down shining up

    # Find the max number of energized tiles
    res = max(shine(map_, wf) for wf in initial_wavefronts)
    return res


def main():
    raw = read_input()
    map_ = parse(raw)

    star1 = shine(map_)
    print(f"Tiles energized: {star1}.")

    star2 = find_highest_energization(map_)
    print(f"A maximum of {star2} tiles can be energized.")


if __name__ == '__main__':
    main()
