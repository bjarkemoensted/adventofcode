import pathlib


def read_input():
    _here = pathlib.Path(__file__).resolve().parent
    fn = _here / "inputs" / "input05.txt"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [int(elem) for elem in s.splitlines()]
    return res


def steps_to_complete(instructions, max_offset=None):
    if max_offset is None:
        max_offset = float('inf')

    instructions = [val for val in instructions]
    ind = 0
    n = 0
    while True:
        try:
            val = instructions[ind]
            shift = -1 if val >= max_offset else +1
            instructions[ind] += shift
            ind += val
            n += 1
        except IndexError:
            return n


def solve(data: str):
    instructions = parse(data)

    star1 = steps_to_complete(instructions)
    print(f"Solution to part 1: {star1}")

    star2 = steps_to_complete(instructions, max_offset=3)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    from aoc.utils.data import check_examples
    check_examples(year=2017, day=5, solver=solve)
    raw = read_input()
    solve(raw)


if __name__ == '__main__':
    main()
