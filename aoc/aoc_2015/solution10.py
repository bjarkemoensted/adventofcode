def extend_sequence(s, n_times=1):
    """Extends a sequence, e.g. transforms "11" into "21" (two ones)."""

    if n_times == 0:
        return s

    res = ""
    buffer = ""

    for char in s:
        if buffer and char != buffer[-1]:
            res += str(len(buffer))+buffer[-1]
            buffer = ""
        buffer += char

    if buffer:
        res += str(len(buffer)) + buffer[-1]

    return extend_sequence(res, n_times - 1)


def solve(data: str):
    extended = data
    for _ in range(40):
        extended = extend_sequence(extended)

    star1 = len(extended)
    print(f"Solution to part 1: {star1}")

    for _ in range(10):
        extended = extend_sequence(extended)
    star2 = len(extended)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 10
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
