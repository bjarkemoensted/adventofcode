from collections import Counter


def read_input():
    with open("input06.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [list(line) for line in s.split("\n")]
    return res


def main():
    raw = read_input()
    parsed = parse(raw)

    # List of characters at positions 0, 1, etc
    transposed = list(map(list, zip(*parsed)))

    most_common_chars = []
    least_common_chars = []
    for chars in transposed:
        counts = Counter(chars)
        chars_by_frequency = sorted(counts.keys(), key=lambda k: counts.get(k, 0))
        most_common_chars.append(chars_by_frequency[-1])
        least_common_chars.append(chars_by_frequency[0])

    solution1 = "".join(most_common_chars)
    print(f"Solution 1: {solution1}.")

    solution2 = "".join(least_common_chars)
    print(f"Solution 2: {solution2}.")


if __name__ == '__main__':
    main()
