import numba
import pathlib


def read_input():
    _here = pathlib.Path(__file__).resolve().parent
    fn = _here / "inputs" / "input17.txt"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = int(s)
    return res


def fill_buffer(steps: int, n_elems=2017):
    buffer = [0]
    pos = 0
    for n in range(1, n_elems + 1):
        pos = (pos + steps) % len(buffer)
        buffer = buffer[:pos+1] + [n] + buffer[pos+1:]
        pos += 1

    res = buffer[(pos + 1) % len(buffer)]

    return res


@numba.njit
def value_at_pos(steps: int, n_elems: int, ind: int):
    res = None
    len_ = 1
    pos = 0
    for n in range(1, n_elems + 1):
        pos = (pos + steps) % len_
        len_ += 1
        pos += 1
        if pos == ind:
            res = n
        #

    return res


def solve(data: str):
    steps = parse(data)

    star1 = fill_buffer(steps)
    print(f"Solution to part 1: {star1}")

    star2 = value_at_pos(steps, n_elems=50_000_000, ind=1)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    from aoc.utils.data import check_examples
    check_examples(year=2017, day=17, solver=solve)
    raw = read_input()
    solve(raw)


if __name__ == '__main__':
    main()
