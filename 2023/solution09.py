def read_input():
    with open("input09.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = [[int(elem) for elem in line.split()] for line in s.split("\n")]
    return res


def extrapolate(numbers):
    if all(n == 0 for n in numbers):
        return 0

    diffs = [numbers[i] - numbers[i-1] for i in range(1, len(numbers))]
    return numbers[-1] + extrapolate(diffs)


def main():
    raw = read_input()
    data = parse(raw)

    star1 = sum(extrapolate(numbers) for numbers in data)
    print(f"Sum of the OASIS extrapolations thingies is {star1}.")

    star2 = sum(extrapolate(numbers[::-1]) for numbers in data)
    print(f"Sum of the OASIS reverse extrapolations is {star2}.")


if __name__ == '__main__':
    main()
