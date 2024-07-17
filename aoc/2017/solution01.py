import pathlib


def read_input():
    _here = pathlib.Path(__file__).resolve().parent
    fn = _here / "inputs" / "input01.txt"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [int(char) for char in s]
    return res


def match_next(arr):
    elems = [n for i, n in enumerate(arr) if arr[(i+1) % len(arr)] == n]
    return elems


def match_halfway(arr):
    elems = [n for i, n in enumerate(arr) if arr[(i+len(arr)//2) % len(arr)] == n]
    return elems


def solve(data: str):
    parsed = parse(data)

    star1 = sum(match_next(parsed))
    print(f"Solution to part 1: {star1}")

    star2 = sum(match_halfway(parsed))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    from aoc.utils.data import check_examples
    check_examples(year=2017, day=1, solver=solve, suppress_output=True)
    raw = read_input()
    solve(raw)


if __name__ == '__main__':
    main()