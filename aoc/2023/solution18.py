def read_input():
    with open("input18.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


directions = dict(
    U=(-1, 0),
    R=(0, 1),
    D=(1, 0),
    L=(0, -1),
)


def parse(s):
    res = []
    for line in s.split("\n"):
        a, b, c = line.split()
        b = int(b)
        c = c[1:-1]
        res.append((a, b, c))

    return res


def walk(crd, dir_, n_steps):
    res = tuple(x + n_steps*delta for x, delta in zip(crd, dir_))
    return res


def decypher_instructions(instructions):
    """Decyphers the hidden instructions for part 2"""

    res = []
    dir_map = {"0": "R", "1": "D", "2": "L", "3": "U"}
    for _, _, hexcode in instructions:
        dir_letter = dir_map[hexcode[-1]]

        hexstring = "0x"+hexcode[1:-1]
        n_steps = int(hexstring, 0)

        instruction = (dir_letter, n_steps)
        res.append(instruction)

    return res


def compute_points(instructions):
    """Determines the points that make up the corners of the lagoon.
    Starts at origin, then repeatedly follows the instructions by taking a specified number of steps
    in the specified direction."""

    current = (0, 0)
    res = [current]
    for dir_letter, n_steps in instructions:
        dir_ = directions[dir_letter]
        current = walk(crd=current, dir_=dir_, n_steps=n_steps)
        res.append(current)

    return res


def compute_area(points):
    """Computes the area contained by the boundary specified by the points using the shoelace formula with Pick's
    theorem to find the number of interior points from the boundary."""

    shoelace = 0
    boundary = 0

    for i, p1 in enumerate(points[:-1]):
        # Add the determinant of p2 and p1 to the shoelace running sum
        (y1, x1) = p1
        p2 = points[i + 1]
        y2, x2 = p2
        shoelace += (y2 * x1 - y1 * x2)

        # Add p1 -> p2 to the boundary
        boundary += sum(abs(x2 - x1) for x1, x2 in zip(p1, p2))

    # Compute area with shoelace formula
    A_shoelace = abs(shoelace // 2)
    # Get the number of interior points (inner area) from Pick's theorem
    inner = A_shoelace - boundary//2 + 1

    res = inner + boundary

    return res


def main():
    raw = read_input()
    instructions_all = parse(raw)

    instructions1 = [(dir_letter, n_steps) for dir_letter, n_steps, _ in instructions_all]
    points = compute_points(instructions1)
    star1 = compute_area(points)
    print(f"The lava lagoon can hold {star1} cubic meters of lava.")

    instructions2 = decypher_instructions(instructions_all)
    points2 = compute_points(instructions2)
    star2 = compute_area(points2)
    print(f"The lava lagoon with the decyphered instructions can hold {star2} cubic meters of lava.")


if __name__ == '__main__':
    main()
