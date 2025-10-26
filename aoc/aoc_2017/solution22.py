# .· `·. .·      `·   *  .  .`·          ·`*.  · ` +  `·.   .· +`·     ·.` *.  `
# ` * . ·  .·   ·•` *  .·  ·     Sporifica Virus · .     ·*`·  ·  +  .  ` · · `.
# ·`. *·.`   ·     · ` https://adventofcode.com/2017/day/22   ·`.     ·`* +`.·*·
# * ·· `•·* .  ` ·. · .+  ·*`·    +. ·` .    ·*    ·    ·+·   `.·   *·   *`.+`· 


import numpy as np


def parse(s: str):
    """Parse the characters to a numpy array"""
    m = np.array([list(line) for line in s.splitlines()])
    return m


def numpy_to_xy(m):
    """Takes a numpy array and returns a set of the x, x coordinates representing an infected node ("#").
    Coordinates are centered so the coordinate at the middle becomes x, y = 0, 0."""

    ioff, joff = (lim // 2 for lim in m.shape)

    res = set([])
    for i, j in np.ndindex(*m.shape):
        x = j - joff
        y = -i + ioff
        if m[i, j] == "#":
            res.add((x, y))
        #
    return res


def display(states, pos):
    """Helper method to display a state similar to on the web page"""
    coords = list(states.keys()) + [pos]
    offset = max(abs(coord) for tup in coords for coord in tup)
    size = 2*offset + 1
    m = np.array([[" . " for _ in range(size)] for _ in range(size)])

    for i, j in np.ndindex(size, size):
        x = j - offset
        y = -i + offset
        sym = states.get((x, y), ".")
        val = f"[{sym}]" if (x, y) == pos else f" {sym} "
        m[i, j] = val

    s = "\n".join(["".join(row) for row in m])
    print(s)


def tick(infected, pos=(0, 0), n_bursts=10_000, evolved=False, show_final_state=False):
    """Simulates n bursts of activity of the worm. evolved denotes whether we're considering the more sophisticated
    worm of part 2."""

    # Use ints instead of the full string
    stages = ["clean", "weakened", "infected", "flagged"] if evolved else ["clean", "infected"]
    infected_ind = stages.index("infected")

    states = {coord: infected_ind for coord in infected}

    # Vars for keeping track of position and direction
    x, y = pos
    dx, dy = 0, 1

    n_transmissions = 0

    for _ in range(n_bursts):
        # Figure out the state of the current node
        current_node = (x, y)
        state_ind = states.get(current_node, 0)
        state = stages[state_ind]

        # Figure out the new direction
        if state == "clean":
            # turn left
            dx, dy = -dy, dx
        elif state == "weakened":
            pass  # Maintain direction
        elif state == "infected":
            # turn right
            dx, dy = dy, -dx
        elif state == "flagged":
            dx, dy = -dx, -dy
        else:
            raise ValueError

        # Figure out how to represent the new state (remove if cleaned, otherwise update)
        newstate = stages[(state_ind + 1) % len(stages)]
        if newstate == "clean":
            del states[current_node]
        else:
            states[current_node] = state_ind + 1

        if newstate == "infected":
            n_transmissions += 1

        x += dx
        y += dy

    if show_final_state:
        pos = (x, y)
        d = {coords: stages[i][0].upper() for coords, i in states.items()}
        d = {k: "#" if v == "I" else v for k, v in d.items()}
        display(d, pos)

    return n_transmissions


def solve(data: str) -> tuple[int|str, int|str]:
    m = parse(data)

    infected = numpy_to_xy(m)
    pos = (0, 0)

    star1 = tick(infected, pos)
    print(f"Solution to part 1: {star1}")

    star2 = tick(infected, pos, n_bursts=10_000_000, evolved=True)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 22
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()