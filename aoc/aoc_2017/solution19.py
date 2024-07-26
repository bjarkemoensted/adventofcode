import numpy as np
import pathlib
from string import ascii_uppercase


def read_input():
    _here = pathlib.Path(__file__).resolve().parent
    fn = _here / "inputs" / "input19.txt"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    lists = [list(line) for line in s.splitlines()]
    res = np.array(lists)

    return res


directions = {
    (1, 0): "|",
    (0, 1): "-",
    (-1, 0): "|",
    (0, -1): "-"
}

steps = {k: np.array(k) for k in directions.keys()}


def trace_path(map_):
    rows, cols = map_.shape

    visited = []
    pos = None
    prev = None
    dir_ = (1, 0)

    for j in range(cols):
        if map_[0, j] == "|":
            pos = (0, j)
            break
        #

    def get_neighbors(coord):
        """Given a coordinate, iterate over the neighboring coordinates, and the direction in which they lie"""
        for vec in directions:
            nc = tuple(x + delta for x, delta in zip(coord, vec))
            if all(0 <= x < lim for x, lim in zip(nc, map_.shape)):
                yield nc, vec
            #
        #

    while True:
        char = map_[pos]
        if char == " ":
            break
        visited.append(char)

        if char == "+":
            # Check possible neighbors and new directions
            candidates = [(site, newdir) for site, newdir in get_neighbors(pos) if site != prev and map_[site] != " "]
            if len(candidates) != 1:
                raise ValueError(candidates, pos)
            # Update both position and direction
            newpos, dir_ = candidates[0]
        else:
            # Just step forward
            newpos = tuple(x + delta for x, delta in zip(pos, dir_))

        prev = pos
        pos = newpos

    return visited


def solve(data: str):
    map_ = parse(data)

    chars = trace_path(map_)
    star1 = "".join(let for let in chars if let in set(ascii_uppercase))
    print(f"Solution to part 1: {star1}")

    star2 = len(chars)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    from aoc.utils.data import check_examples
    check_examples(year=2017, day=19, solver=solve)
    raw = read_input()
    solve(raw)


if __name__ == '__main__':
    main()
