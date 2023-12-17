import numpy as np


def read_input():
    with open("input17.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    M = np.array([[int(char) for char in line] for line in s.split("\n")])
    return M


north = (-1, 0)
west = (0, -1)
east = (0, 1)
south = (1, 0)

all_dirs = (north, west, south, east)


def update_history(hist, dir_):
    """Updates the history (distance travelled since last direction change).
    Throws an error if the rules are violated."""

    opposite_direction = any(a*b < 0 for a, b in zip(hist, dir_))
    same_direction = any(a*b > 0 for a, b in zip(hist, dir_))
    res = dir_ if not same_direction else tuple(a + b for a, b in zip(hist, dir_))
    # Not allowed to make sudden U-turns or going 4 or more steps in the same direction
    if opposite_direction or any(abs(x) > 3 for x in res):
        raise ValueError

    return res


def update_history_ultra(hist, dir_):
    """Updates history following the rules for ultra crucibles"""
    opposite_direction = any(a*b < 0 for a, b in zip(hist, dir_))
    same_direction = any(a*b > 0 for a, b in zip(hist, dir_))
    # Must have travelled 4 steps before turning. Demanding dist > 0 to allow movement from the initial site
    if not same_direction and 0 < sum(abs(x) for x in hist) < 4:
        raise ValueError
    res = dir_ if not same_direction else tuple(a + b for a, b in zip(hist, dir_))
    # Disallow U-turns and travelling in the same direction for 10 or more steps
    if opposite_direction or any(abs(x) > 10 for x in res):
        raise ValueError

    return res


def take_steps(crd, hist, M, ultra):
    """Takes a coordinate and history. Provides the allowed subsequent coordinates and histories."""
    update_fun = update_history_ultra if ultra else update_history
    for dir_ in all_dirs:
        newcrd = tuple(x + val for x, val in zip(crd, dir_))
        if not all(0 <= x < lim for x, lim in zip(newcrd, M.shape)):
            continue
        try:
            newhist = update_fun(hist, dir_)
        except ValueError:
            continue

        yield newcrd, newhist


def find_best_path(M, destination, ultra=False):
    """Find the path which minimizes the heat loss"""
    initial_state = ((0, 0), (0, 0))
    states = [initial_state]
    state2heatloss = {initial_state: 0}
    best = float("inf")
    done = False
    n = 0
    while not done:
        newstates = []
        for state in states:
            heat_loss = state2heatloss[state]
            crd, hist = state
            for newstate in take_steps(crd, hist, M, ultra=ultra):
                newcrd, newhist = newstate
                new_heat_loss = heat_loss + M[newcrd]
                if state2heatloss.get(newstate, float("inf")) > new_heat_loss:
                    state2heatloss[newstate] = new_heat_loss
                    newstates.append(newstate)
                    arrived = newcrd == destination
                    if ultra:
                        arrived = arrived and sum(map(abs, newhist)) >= 4
                    if arrived:
                        best = min(best, new_heat_loss)
                #
            #
        states = newstates
        done = not states
        n += 1
        print(f"Iteration {n}, considering {len(states)} states. Current best: {best}.", end="\r")
    print()

    return best


def main():
    raw = read_input()
    M = parse(raw)
    destination = tuple(lim - 1 for lim in M.shape)
    star1 = find_best_path(M, destination)
    print(f"Minimum possible heat loss is: {star1}.")

    star2 = find_best_path(M, destination, ultra=True)
    print(f"Minimum possible heat loss with ultra crucibles is: {star2}.")


if __name__ == '__main__':
    main()
