from collections import Counter
import pathlib


def read_input():
    _here = pathlib.Path(__file__).resolve().parent
    fn = _here / "inputs" / "input04.txt"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [line.split() for line in s.splitlines()]
    return res


def passphrase_is_valid(passphrase: list) -> bool:
    c = Counter(passphrase)
    return all(v == 1 for v in c.values())


def solve(data: str):
    parsed = parse(data)

    valid = [elem for elem in parsed if passphrase_is_valid(elem)]
    star1 = len(valid)
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    from aoc.utils.data import check_examples
    check_examples(year=2017, day=4, solver=solve)
    raw = read_input()
    solve(raw)


if __name__ == '__main__':
    main()