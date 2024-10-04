from collections import Counter

def parse(s):
    res = [list(line) for line in s.split("\n")]
    return res


def solve(data: str):
    parsed = parse(data)

    # List of characters at positions 0, 1, etc
    transposed = list(map(list, zip(*parsed)))

    most_common_chars = []
    least_common_chars = []
    for chars in transposed:
        counts = Counter(chars)
        chars_by_frequency = sorted(counts.keys(), key=lambda k: counts.get(k, 0))
        most_common_chars.append(chars_by_frequency[-1])
        least_common_chars.append(chars_by_frequency[0])

    star1 = "".join(most_common_chars)
    print(f"Solution to part 1: {star1}")

    star2 = "".join(least_common_chars)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2016, 6
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
