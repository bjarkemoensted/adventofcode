def read_input():
    with open("input03.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = []
    for line in s.split("\n"):
        sides = [int(x) for x in line.split()]
        res.append(sides)

    return res


def triangle_is_possible(sides):
    ordered = sorted(sides)
    return sum(ordered[:-1]) > ordered[-1]


def main():
    raw = read_input()
    parsed = parse(raw)

    n_possible = sum(triangle_is_possible(sides) for sides in parsed)
    print(f"There are {n_possible} possible triangles.")

    new_sides = []
    for ind in range(len(parsed[0])):
        these_sides = [col[ind] for col in parsed]
        for i in range(0, len(these_sides), 3):
            new_sides.append(these_sides[i:i+3])

    n_new = sum(triangle_is_possible(sides) for sides in new_sides)
    print(f"Turns out there's {n_new} possible triangles.")


if __name__ == '__main__':
    main()
