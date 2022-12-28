import numpy as np


def read_input():
    with open("input24.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    """Parses the input into a tuple of start/end locations, and a dict mapping coordinates (i,j) to blizzards."""
    lines = s.split("\n")
    rows = len(lines)
    cols = len(lines[0])
    assert all(len(line) == cols for line in lines)

    start_end = ((0, 1), (rows-1, cols-2))
    assert all(lines[i][j] == "." for i, j in start_end)
    d = {}

    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char in "<>v^":
                d[(i, j)] = [char]

    return start_end, d


direction2unit_vector = {
    ">": (0, 1),
    "<": (0, -1),
    "v": (1, 0),
    "^": (-1, 0),
}


def step(pos, dir_, limits, wrap=True):
    """Changes input position (i, j - tuple) one step in the specified direction (<>v^).
    Automatically wraps around the edges of the map."""
    new_pos = [crd for crd in pos]
    vec = direction2unit_vector[dir_]
    for ind in range(len(new_pos)):
        old_crd = new_pos[ind]
        limit = limits[ind]
        delta = vec[ind]
        shifted = old_crd + delta
        if wrap:
            if shifted >= limit:
                shifted = 1
            elif shifted <= 0:
                shifted = limit - 1
        new_pos[ind] = shifted

    return tuple(new_pos)


def get_new_positions(pos, start, end, limits):
    """Returns all positions reachable (ignoring blizzards) form the input position."""
    res = [pos]
    for dir_ in direction2unit_vector.keys():
        new_pos = step(pos, dir_, limits, wrap=False)
        hit_edge = not all(0 < crd < limit for crd, limit in zip(new_pos, limits))
        if hit_edge and new_pos not in (start, end):
            continue
        else:
            res.append(new_pos)
        #

    return res


def update_blizzards(blizzards, limits):
    """Takes a dict mapping coordinates to blizzards (<>v^) at that location, along with map limits, and returns
    the locations of blizzards at the subsequent time iteration."""

    res = {}
    for pos, directions in blizzards.items():
        for dir_ in directions:
            new_pos = step(pos, dir_, limits)
            try:
                res[new_pos].append(dir_)
            except KeyError:
                res[new_pos] = [dir_]
            #
        #

    return res


def display(pos, blizzards, limits):
    """Prints a current state, for debugging and such."""
    rows, cols = (val + 1 for val in limits)
    M = np.array([["." for _ in range(cols)] for _ in range(rows)])
    start_end = ((0, 1), (rows-1, cols-2))

    for i in range(rows):
        for j in range(cols):
            if (i, j) == pos:
                M[i, j] = "E"
                continue
            if (i, j) in start_end:
                continue
            edge = (i == 0) or (j == 0) or (i == rows - 1) or (j == cols - 1)
            if edge:
                M[i, j] = "#"

            try:
                symbols = blizzards[(i, j)]
                char = str(len(symbols)) if len(symbols) > 1 else symbols[0]
                M[i, j] = char
            except KeyError:
                pass

    print("\n".join(["".join(row) for row in M]))


def manhatten_distance(a, b):
    """Computes Manhatten dist between two points"""
    res = sum([abs(crd1 - crd2) for crd1, crd2 in zip(a, b)])
    return res


def _represent_blizzards(blizzards):
    """Makes a hashabale representation of a configuration of blizzards. Handy for checking if a blizzard state
    has been encountered previously."""

    keys = sorted(blizzards.keys())
    res = tuple((k, tuple(sorted(blizzards[k]))) for k in keys)
    return res


def generate_all_blizzard_states(blizzards, limits):
    """Generates all configurations of blizzards that will occur. Due to the cyclical nature of the blizzards,
    the pattern will repeat indefinitely, so if this method returns, e.g. 20 unique states, any state can be found
    by taking the time elapsed modulo e.g. 20."""

    # Keep track of which blizzard configurations have been encountered
    rep = _represent_blizzards(blizzards)
    seen = {rep}
    # Store all the unique states
    res = [blizzards]
    while True:
        blizzards = update_blizzards(blizzards, limits)
        rep = _represent_blizzards(blizzards)
        if rep in seen:
            break
        else:
            res.append(blizzards)
            seen.add(rep)
        #

    return res


def crunch(start, end, blizzard_states, limits, start_time=0):
    """Finds the shortest path from a to b, keeping the elves clear from any blizzards."""

    cut = 100  # Number of the most promising states to iterate on

    # Create the initial state (located at the starting position, with the specified start time elapsed)
    initial_state = {"pos": start, "time": start_time}

    # Keep an ongoing list of states considered
    iterate_from = [initial_state]

    # Keep track of how long it has taken to reach any state we encounter
    rep = (start, initial_state["time"] % len(blizzard_states))
    state2shortest_time = {rep: 0}

    # Keep running record of the shortest distance found so far. Discard any state that cannot possibly catch up.
    record = float("inf")

    while iterate_from:
        # Partition into the most promising (a) and least promising (b) existing states
        a = iterate_from[:cut]
        b = iterate_from[cut:]

        # List for holding the new states grown in this step
        grown = []

        for old_state in a:
            # Consider any location we may travel to from current position
            new_positions = get_new_positions(old_state["pos"], start, end, limits)
            for new_pos in new_positions:
                # Update time elapsed and the 'blizzard index' which indicates where blizzards are located
                new_time = old_state["time"] + 1
                new_blizzard_ind = new_time % len(blizzard_states)
                new_state = {"pos": new_pos, "time": new_time}
                rep = (new_pos, new_blizzard_ind)

                # Drop the new state if we stepped into a blizzard
                if new_pos in blizzard_states[new_blizzard_ind]:
                    continue

                # If we're at the destination, update the shortest path found if necessary
                if new_pos == end:
                    record = min(record, new_time)
                    continue

                # If we hit the edge of the map, do not iterate from the new state
                at_edge = not all(0 <= crd <= limit for crd, limit in zip(new_pos, limits))
                out_of_bounds = at_edge and (new_pos not in (start, end))
                if out_of_bounds:
                    continue

                # If we've already found a shorter path to the current state, do not iterate further
                if new_state["time"] >= state2shortest_time.get(rep, float("inf")):
                    continue

                # Otherwise, note that this is the fastest way (so far) of getting here
                state2shortest_time[rep] = new_time

                # If we still have a chance to beat the record from here, keep iterating from this state.
                lower_bound = new_time + manhatten_distance(new_pos, end)
                if lower_bound < record:
                    grown.append(new_state)
                #
            #

        # Sort states by distance to target, and repeat iteration until we run out of states to consider.
        grown.sort(key=lambda s: manhatten_distance(s["pos"], end))
        iterate_from = grown + b

    return record


def shortest_path_with_snack_return(start, end, blizzard_states, limits):
    """Finds the shortest path which includes a return trip to pick up forgotten snacks."""
    first_trip_duration = crunch(start, end, blizzard_states, limits, start_time=0)
    return_trip_duration = crunch(end, start, blizzard_states, limits, start_time=first_trip_duration)
    second_trip_duration = crunch(start, end, blizzard_states, limits, start_time=return_trip_duration)

    return second_trip_duration


def main():
    raw = read_input()
    (start, end), blizzards = parse(raw)
    limits = end[0], end[1] + 1
    blizzard_states = generate_all_blizzard_states(blizzards, limits)

    shortest = crunch(start, end, blizzard_states, limits)
    print(f"The shortest path takes {shortest} minutes.")

    shortest_w_snaccs = shortest_path_with_snack_return(start, end, blizzard_states, limits)
    print(f"Shortest route including a return for the forgotten snacks takes {shortest_w_snaccs} minutes.")


if __name__ == '__main__':
    main()
