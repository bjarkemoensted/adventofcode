import numba
import pathlib
import re


def read_input():
    _here = pathlib.Path(__file__).resolve().parent
    fn = _here / "inputs" / "input15.txt"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = dict()
    pat = r"Generator (?P<name>\S+) starts with (?P<value>\d+)"
    for line in s.splitlines():
        m = re.match(pat, line)
        d = m.groupdict()
        res[d["name"].lower()] = int(d["value"])
    return res


def generate(seed: int, factor: int, mod=None):
    div = 2147483647
    val = seed

    while True:
        val = (val*factor) % div
        yield val
    pass


@numba.jit
def count_matches(a, b, n_its=40_000_000, a_mod=1, b_mod=1) -> int:
    res = 0
    div = 2147483647
    db = 2**16
    for _ in range(n_its):
        a = a * 16807 % div
        while a % a_mod != 0:
            a = a * 16807 % div

        b = b * 48271 % div
        while b % b_mod != 0:
            b = b * 48271 % div

        res += a % db == b % db

    return res


def solve(data: str):
    seeds = parse(data)

    star1 = count_matches(**seeds)
    print(f"Solution to part 1: {star1}")

    star2 = count_matches(**seeds, n_its=5_000_000, a_mod=4, b_mod=8)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    from aoc.utils.data import check_examples
    check_examples(year=2017, day=15, solver=solve)
    raw = read_input()
    solve(raw)


if __name__ == '__main__':
    main()
