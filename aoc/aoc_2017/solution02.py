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
    year, day = 2017, 2
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()