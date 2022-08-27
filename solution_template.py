def read_input():
    with open("{INPUT_FILENAME}") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = s  # TODO parse input here
    return res


def main():
    raw = read_input()
    parsed = parse(raw)

    # TODO solve puzzle


if __name__ == '__main__':
    main()
