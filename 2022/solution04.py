def read_input():
    with open("input04.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    tups = []
    for line in s.strip().split():
        vals = [tuple(int(x) for x in part.split("-")) for part in line.strip().split(",")]
        tups.append(vals)

    return tups


def tuple_contains_other(tup1, tup2):
    a1, b1 = tup1
    a2, b2 = tup2

    return a1 <= a2 and b1 >= b2


def tuple_have_overlap(tup1, tup2):
    a1, b1 = tup1
    a2, b2 = tup2

    return b1 >= a2 and not a1 > b2


def main():
    raw = read_input()
    parsed = parse(raw)

    n_contained = 0
    n_overlaps = 0
    for tup1, tup2 in parsed:
        n_contained += tuple_contains_other(tup1, tup2) or tuple_contains_other(tup2, tup1)
        n_overlaps += tuple_have_overlap(tup1, tup2) or tuple_have_overlap(tup2, tup1)

    print(f"Found {n_contained} sections contained in other sections.")

    print(f"Found {n_overlaps} overlapping sections.")


if __name__ == '__main__':
    main()
