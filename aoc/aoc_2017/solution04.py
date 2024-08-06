from collections import Counter


def parse(s):
    res = [line.split() for line in s.splitlines()]
    return res


def passphrase_is_valid(passphrase: list, include_anagrams=False) -> bool:
    elems = [''.join(sorted(part)) for part in passphrase] if include_anagrams else passphrase
    c = Counter(elems)

    return all(v == 1 for v in c.values())


def solve(data: str):
    parsed = parse(data)

    valid = [elem for elem in parsed if passphrase_is_valid(elem)]
    star1 = len(valid)
    print(f"Solution to part 1: {star1}")

    star2 = sum(passphrase_is_valid(phrase, include_anagrams=True) for phrase in parsed)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 4
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()