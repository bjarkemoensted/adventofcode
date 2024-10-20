# .+. ⸳`⸳ꞏ ` ꞏ.•  *  . `⸳.*      .⸳ ꞏ•  *ꞏ.⸳     ⸳`.ꞏ ⸳`*  .. •ꞏ ⸳ `. +.⸳  *   `
#    ..•ꞏ `ꞏ * `  •⸳ꞏꞏ* ` .`   ⸳  Particle Swarm +.`    `  ꞏ ` .⸳+ ꞏ⸳.   *.` • ꞏ
#  .ꞏ⸳ꞏ*.   ⸳     .`*⸳ https://adventofcode.com/2017/day/20    ꞏ⸳. ⸳   ꞏ` *ꞏ  ⸳.
# ⸳ *`⸳ꞏ+     •.  ` .•   ꞏ`⸳• ꞏ*`⸳ .  ⸳ꞏ*.ꞏ`        `.•. *⸳ ⸳ꞏ  .`ꞏ *   ⸳  .ꞏ.*•


from collections import defaultdict
import numba
import re


def parse(s):
    res = []
    for line in s.splitlines():
        hits = re.findall(r"<(.*?)>", line)
        p, v, a = [tuple(map(int, part.split(","))) for part in hits]
        res.append((p, v, a))
    return res


def get_orders(state):
    """Takes a state where each component is a tuple of the zeroth, firsth, and second order parts of the position.
    Changes sign so the dominant (highest order) has positive sign, then sums the contributions for each order.
    The result is a 3-tuple where the 0th element is the sum of second order terms, 1st element the sum of the first
    order terms, and the last element is the sum of the constant terms.
    Ordering states by the resulting tuple results in the same ordering as the asymptotic (t -> inf) Manhatten
    distance to origo. The logic is that by default, the acceleration (second order) term will dominate the reulst,
    but if a tie-breaker is needed, the velocity component will be used, but with opposite sign to the acceleration,
    and so on."""

    mags = [0.0 for _ in state]
    for component in state:
        avp = component[::-1]
        dominant_sign = None
        for i, elem in enumerate(avp):
            if elem == 0:
                continue
            if dominant_sign is None:
                dominant_sign = +1 if elem > 0 else -1
            mags[i] += dominant_sign*elem

    res = tuple(mags)

    return res


def nearest_longterm(states):
    """Determine the state which will asymptotically be closest to origo (Manhatten dist)"""
    res = min(range(len(states)), key=lambda i: get_orders(states[i]))

    return res


def compute_position_at_time(parametrized_state, t):
    """Takes a state (3-tuple of const, first, and second order contributions to position for each component (x, y, z)),
    and returns the position (3-tuple) at time t"""
    res = tuple(c + b*t + a*t**2 for c, b, a in parametrized_state)
    return res


def _parametrize_state(state):
    """Converts a state from the format given in the problem (position, velocity, and acceleration for each dimension),
    to a format which is easier to use to compute position at arbitrary times.
    Since v = x', and a = x'', we can express each coordinate as a function of time.
    Because the problem uses discrete time, the second order (acceleration) term isn't a/2*t^2, but rather
    a/2 * t*(t-1), which can be found using the Gauss summation thingy.

    For each coordinate x, we get 0th, 1st, and 2nd order terms like:
    x(t) = x0 + v0*t + a0*t*(t + 1) // 2
         = x0 + (v0 + a0//2)*t + a0//2 * t^2
    """

    res = []
    for px, vx, ax in zip(*state):
        # c, b, a
        quad = (px, vx + ax / 2, ax / 2)
        res.append(quad)

    res = tuple(res)
    return res


@numba.njit
def solve_quad(c, b, a):
    """Solves a quadratic equation c + b*t + a^t = 0,  with the provided values.
    Returns a list of solutions, which will hold 0, 1, or 2 elements."""

    # Ugly hack to signal to numba which data type the array will hold
    res = [float(val) for val in range(0)]

    if a == 0:
        # First-order equations c + b*t = 0 simply give t = -c/b
        if b != 0:
            res = [-c/b]
        #
    else:
        # If there's a second order term, compute the solutions, if any
        d = b**2 - 4*a*c
        if d > 0:
            sqrt = d ** 0.5
            res = [(-b + sig * sqrt) / (2 * a) for sig in (+1, -1)]
        elif d == 0:
            res = [-b / (2 * a)]
        else:
            pass
        #

    return res
    #


def determine_time_of_collision(state_a, state_b):
    """Takes 2 states, and returns the time at which a collision between them occurs.
    In case this happens at multiple times, the first time is returned.
    If no collision will occur, None is returned."""

    solutions = []

    # Iterate over each direction for the pair of states
    for xa, xb in zip(state_a, state_b):

        # Compute the difference along the current direction between the two states
        diffs = tuple(ca - cb for ca, cb in zip(xa, xb))
        if all(diff == 0 for diff in diffs):
            continue  # If they're exactly the same in this direction, we can't get any information from the current dir

        # Attempt to determine possible collision times
        c, b, a = diffs

        # Only look for integer solutions
        if not solutions:
            solutions = [round(t) for t in solve_quad(c, b, a)]
        solutions = [t for t in solutions if c + b * t + a * t ** 2 == 0]

        # If there's no solution for this direction, the trajectories will never intersect, so no collision
        if len(solutions) == 0:
            return None
        #

    # If there's 2 collision times, keep only the first, as the pair will be destroyed after the first collision
    res = min(solutions)
    return res


def count_remaining(states):
    """Counts the number of states remaining after all collisions have been resolved"""

    # Map timestamps (n_ticks) to list of pairs of states which collide at that time
    collisions = defaultdict(lambda: [])

    # Check each pair of states
    for i, state_a in enumerate(states):
        for j in range(i+1, len(states)):
            state_b = states[j]

            # If a and b collide, note the time of the collision
            t_c = determine_time_of_collision(state_a, state_b)
            if t_c is not None:
                collisions[t_c].append((i, j))
            #
        #

    # Go over the collisions in the order they occur, and remove states as they're destroyed in collisions
    remaining = {i for i in range(len(states))}
    for tick, colliding_pairs in sorted(collisions.items()):
        # Check which states are destroyed in collisions at current timestamp
        destroyed = set([])
        for pair in colliding_pairs:
            # Skip collisions where either state has previously been destroyed
            if all(state_ind in remaining for state_ind in pair):
                i, j = pair
                destroyed.add(i)
                destroyed.add(j)
            #
        remaining -= destroyed

    res = len(remaining)
    return res


def solve(data: str):
    states_raw = parse(data)
    states = [_parametrize_state(state) for state in states_raw]

    star1 = nearest_longterm(states = states)
    print(f"Solution to part 1: {star1}")

    star2 = count_remaining(states)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 20
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
