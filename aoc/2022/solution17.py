def read_input():
    with open("input17.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


shapes_chars = """####

.#.
###
.#.

..#
..#
###

#
#
#
#

##
##"""


test = """>>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>"""

def parse_shapes():
    res = []
    chunks = shapes_chars.split("\n\n")
    for chunk in chunks:
        occupied = set([])
        lines = chunk.split("\n")
        for i, line in enumerate(lines):
            y = len(lines) - i
            for x, char in enumerate(line):
                if char == "#":
                    occupied.add((x, y))
                #
            #
        res.append(occupied)

    return res


def parse(s):
    res = list(s)
    return res


def display(occupied, shape=None):
    import numpy as np

    total_occ = occupied.union(shape)
    rows = max(y for _, y in total_occ) + 1
    cols = max(x for x, _ in total_occ) + 1

    M = np.array([["." for _ in range(cols)] for _ in range(rows)])

    for x, y in occupied:
        M[y, x] = "#"

    if shape:
        for x, y in shape:
            M[y, x] = "@"

    lines = []
    for row in np.flip(M, 0):
        lines.append("".join(row))
    print("\n".join(lines))
    print()


def _leftmost(shape):
    return min(x for x, _ in shape)


def _rightmost(shape):
    return max(x for x, _ in shape)


def _lowest(shape):
    return min(y for _, y in shape)


def _highest(shape):
    return max(y for _, y in shape)


def _shift(coords, step):
    x, y = coords
    if step == ">":
        x += 1
    elif step == "<":
        x -= 1
    elif step == "v":
        y -= 1
    else:
        raise ValueError

    return x, y


def move(shape, step):
    return {_shift(coords, step) for coords in shape}


def _shift_to_starting_position(shape, highest_point, x_offset=2, y_offset=3):
    y_offset += 1
    left_edge = _leftmost(shape)
    lower_edge = _lowest(shape)
    translated = set([])
    for x, y in shape:
        xprime = x - left_edge + x_offset
        yprime = highest_point + y_offset - lower_edge + y
        translated.add((xprime, yprime))

    assert _leftmost(translated) == x_offset and _lowest(translated) - highest_point == y_offset

    return translated


class Cycle:
    def __init__(self, iterable):
        self.n_yielded = 0
        self.ind = 0
        self.stuff = [val for val in iterable]

    def __next__(self):
        self.n_yielded += 1
        res = self.stuff[self.ind]
        self.ind = (self.ind + 1) % len(self.stuff)
        return res


def _signature(occupied, width=7):
    """Returns a 'signature' for a given configuration. Considering the highest point at each value of x,
    this returns a tuple of the height above the lowest point at each location.
    For example, if the highest points are [20, 21, 22] this will return (0, 1, 2)."""

    xvals = list(range(width))

    top = []
    for xval in xvals:
        top.append(max(y for x, y in occupied if x == xval))

    lowest = min(top)
    top = tuple(val - lowest for val in top)
    return top


def represent_state(occupied, shape_generator, step_generator):
    """Returns a canonical, hashable representation of a simulation state."""
    sig = _signature(occupied)
    state = (shape_generator.ind, step_generator.ind, sig)
    return state


def simulate_single_fall(occupied, shape, dir_cycle, width=7, verbose=False):
    """Simulates a single rock falling. Shape specifies the shape of the falling rock, and
    dir_cycle is a generator type iterable (works with next()) that provides the wind directions."""

    # Shift the rock to its starting position
    highest_point = max(y for _, y in occupied)
    shape = _shift_to_starting_position(shape, highest_point)
    if verbose:
        display(occupied, shape)

    falling = True
    while falling:
        # Update position due to lava jet pushing the rock around
        step = next(dir_cycle)
        updated = move(shape, step)
        hit_wall = _leftmost(updated) < 0 or _rightmost(updated) >= width
        collision = len(updated.intersection(occupied)) > 0

        # Update position if the rock doesn't collide with a wall or another rock
        if not (hit_wall or collision):
            shape = updated

        if verbose:
            print(f"Lava goes {step}:")
            display(occupied, shape)

        # Update position due to gravity
        updated = move(shape, "v")
        landed = len(updated.intersection(occupied)) > 0
        if landed:
            return shape
        else:
            shape = updated
            if verbose:
                print(f"Gravity goes v:")
                display(occupied, shape)


def simulate(shapes, directions, width=7, n_shapes=2022):
    """Run a simulation of n_shapes rocks falling."""

    # Define the floor and cycles for shapes and wind directions
    occupied = {(x, 0) for x in range(width)}
    shapes_cycle = Cycle(shapes)
    dir_cycle = Cycle(directions)

    for n in range(n_shapes):
        shape = next(shapes_cycle)
        landed = simulate_single_fall(occupied, shape, dir_cycle)
        occupied.update(landed)

    return occupied


def compute_height_after_n_falls(shapes, directions, width=7, n=1000000000000):
    """Computes tower height after a large number of rocks have fallen by exploiting the cyclical nature or
    the falling shapes and the wind directions."""

    # Maintain a mapping of simulation state to tower height and n rocks dropped
    state2height_n = {}

    # Define the floor and shape+wind direction cycles
    occupied = {(x, 0) for x in range(width)}
    shapes_cycle = Cycle(shapes)
    dir_cycle = Cycle(directions)

    recurred = False
    n_dropped = 0

    # Add the beginning state
    initial_state = represent_state(occupied, shape_generator=shapes_cycle, step_generator=dir_cycle)
    state2height_n[initial_state] = (_highest(occupied), n_dropped)

    while not recurred:
        # Drop a stone
        shape = next(shapes_cycle)
        landed = simulate_single_fall(occupied, shape, dir_cycle)
        occupied.update(landed)
        n_dropped += 1

        # Check if state has been encountered before
        state = represent_state(occupied, shape_generator=shapes_cycle, step_generator=dir_cycle)
        if state in state2height_n:
            recurred = True
        else:
            state2height_n[state] = (_highest(occupied), n_dropped)

    # Determine how much the tower has grown since we last saw the recurring state
    height_first, n_first = state2height_n[state]
    height_second = _highest(occupied)
    delta_height = height_second - height_first

    # Find the average growth rate per rock dropped
    delta_n = n_dropped - n_first
    # Scale up the tower height by the number of times we'll see this state before hitting N rocks
    n_remaining = n - n_first
    n_loops = n_remaining // delta_n
    loops_height = (n_loops)*delta_height

    # Keep dropping rocks until we reach the exact value of n desired
    n_remaining = n_remaining % n_loops
    for _ in range(n_remaining):
        shape = next(shapes_cycle)
        landed = simulate_single_fall(occupied, shape, dir_cycle)
        occupied.update(landed)

    # Compute the final height
    height_third = _highest(occupied)
    total_height = height_first + loops_height + (height_third - height_second)

    return total_height


def main():
    raw = read_input()
    directions = parse(raw)
    shapes = parse_shapes()

    end_state = simulate(shapes, directions)
    print(f"The resulting tower has height {_highest(end_state)}.")

    n = 1000000000000
    final_height = compute_height_after_n_falls(shapes, directions, n=n)
    print(f"Tower height after {n} rocks have fallen is {final_height}.")


if __name__ == '__main__':
    main()
