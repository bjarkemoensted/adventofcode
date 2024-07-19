import pathlib


def read_input():
    _here = pathlib.Path(__file__).resolve().parent
    fn = _here / "inputs" / "input02.txt"
    with open(fn) as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [[int(elem) for elem in line.split()] for line in s.splitlines()]
    return res


def checksum(data):
    diffs = [max(nums) - min(nums) for nums in data]
    res = sum(diffs)
    return res


def checksum2(data):
    def find_div(numbers):
        numbers = sorted(numbers)
        for i, small in enumerate(numbers):
            for large in numbers[i+1:]:
                if large % small == 0:
                    return large // small
            #
        return 0

    res = sum(find_div(numbers) for numbers in data)
    return res


def solve(data: str):
    parsed = parse(data)

    star1 = checksum(parsed)
    print(f"Solution to part 1: {star1}")

    star2 = checksum2(parsed)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    from aoc.utils.data import check_examples
    check_examples(year=2017, day=2, solver=solve, suppress_output=False)

    raw = read_input()
    solve(raw)


if __name__ == '__main__':
    main()