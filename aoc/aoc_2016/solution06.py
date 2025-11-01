# •··  `  ·*.·    · `  ·    *`·  .`·*·       ·.  ·`*.+   ·` .    ·* ·.  ·  *.+·`
# ·` . *·  `  ··*.  ·+   ·` .·  Signals and Noise ·. `*·  ·.`   *   `·· .`+ ·` ·
# .   ·· `+·   `··   ` https://adventofcode.com/2016/day/6      ·` +·  ·*.··  .*
#  . ·  *·  ·• +.  ·*·  .` ·   ·•   `+·   ·  `· . *+·      · · `•  · +` ·· . *`·


from collections import Counter


def parse(s: str):
    res = [list(line) for line in s.split("\n")]
    return res


def solve(data: str) -> tuple[int|str, int|str]:
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


def main() -> None:
    year, day = 2016, 6
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
