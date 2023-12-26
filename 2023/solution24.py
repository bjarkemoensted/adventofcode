import sympy


def read_input():
    with open("input24.txt") as f:
        puzzle_input = f.read()

    return puzzle_input


def parse(s):
    """Parses into format [[x0, y0, z0, vx, vy, vz], ...]"""
    res = []
    for line in s.split("\n"):
        parts = line.split(" @ ")
        vecs = [[int(elem) for elem in part.split(", ")] for part in parts]
        res.append(sum(vecs, []))

    return res


def count_xy_collisions_in_box(hails, space_limit):
    """Counts the number of pairs of hailstones whose future paths will collide in the provided box in the xy-plane."""

    low, high = space_limit
    res = 0

    # Define variables to work with
    temporal = sympy.symbols("ta tb")
    ta, tb = temporal
    spatial = sympy.symbols("xa xb ya yb")
    xa, xb, ya, yb = spatial
    velocity = sympy.symbols("vxa, vxb, vya, vyb")
    vxa, vxb, vya, vyb = velocity

    # Express x and y for hailstones a and b
    eq_xa = xa + vxa*ta
    eq_xb = xb + vxb*tb
    eq_ya = ya + vya * ta
    eq_yb = yb + vyb * tb

    # Generate functions to get the coordinates (x, y) for hail A (looking at path intersection, so just check one)
    coord_funs = [sympy.lambdify(spatial + velocity + temporal, expr) for expr in (eq_xa, eq_ya)]

    # Algebraically solve for the times ta and tb, where the hailstone paths intersect in the xy-plane
    solved_for_t = sympy.solve(
        [
            sympy.Eq(eq_xa, eq_xb),
            sympy.Eq(eq_ya, eq_yb),
        ],
        ta, tb
    )

    # Generate functions to compute ta and tb given the remaining variables (much faster than substituting values)
    maps = dict()
    for var in temporal:
        expr = solved_for_t[var]
        maps[var] = sympy.lambdify(spatial+velocity, expr)

    for i, a in enumerate(hails[:-1]):
        xha, yha, _, vxha, vyha, _ = a
        for b in hails[i+1:]:
            xhb, yhb, _, vxhb, vyhb, _ = b

            # Combine the values for each hailstone into a single dict
            vals = dict(xa=xha, xb=xhb, ya=yha, yb=yhb, vxa=vxha, vxb=vxhb, vya=vyha, vyb=vyhb)
            try:
                t = dict(ta=maps[ta](**vals), tb=maps[tb](**vals))
            except ZeroDivisionError:
                continue  # Happens if trajectories are parallel, so just ignore these

            # If the intersection is in the future and inside the box, count +1 collision.
            in_future = all(val >= 0 for val in t.values())
            all_vals = vals | t
            inside_box = all(low <= f(**all_vals) <= high for f in coord_funs)

            res += inside_box and in_future

    print()

    return res


def solve_throw(hails):
    """Solve for the initial position and velocity needed to hit all hailstones. Return the sum of each component
    of the initial position."""

    xr, yr, zr, vxr, vyr, vzr = sympy.symbols("xr yr zr vxr vyr vzr")
    equations = []
    for hail in hails:
        xh, yh, zh, vxh, vyh, vzh = hail

        # Found these by setting rock and hail positions equal and solving for t in the x, y, and z dimensions
        xy_intersect = (xr - xh)*(vyr - vyh) - (yr - yh)*(vxr - vxh)
        yz_intersect = (yr - yh)*(vzr - vzh) - (zr - zh)*(vyr - vyh)

        equations.append(xy_intersect)
        equations.append(yz_intersect)

    # Check that a single unique solution is found
    solutions = sympy.solve(equations)
    assert len(solutions) == 1

    solution = solutions[0]
    res = sum(solution[k] for k in (xr, yr, zr))

    return res


def main():
    raw = read_input()
    hails = parse(raw)

    box = (200000000000000, 400000000000000)

    star1 = count_xy_collisions_in_box(hails, space_limit=box)
    print(f"In total, {star1} paths collide in the test box.")

    star2 = solve_throw(hails)
    print(f"The rock's x, y, and z coordinates sum to: {star2}.")


if __name__ == '__main__':
    main()

