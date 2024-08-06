def parse(s, as_ascii_codes=False):
    if as_ascii_codes:
        res = [ord(char) for char in s]
        standard_length_suffixes = [17, 31, 73, 47, 23]
        res += standard_length_suffixes
    else:
        res = []
        for elem in s.strip().split(","):
            # Treading carefully bc the data for examples part 2 won't parse as ints
            try:
                res.append(int(elem))
            except ValueError:
                pass
    return res


def emulate_knot(lengths, n: int, n_repititions: int = 1):
    """Simulate the knot hashing thing."""
    pos = 0
    vals = list(range(n))
    skip_size = 0

    for _ in range(n_repititions):
        for len_ in lengths:
            # Wrapping is hard, so just make a temp list shifted by current position, then shift back later
            temp = vals[pos:] + vals[:pos]
            # Reverse the selected list (no problem with wrapping bc the selection starts at index 0 now)
            temp[:len_] = temp[:len_][::-1]
            # Shift back
            vals = temp[-pos:] + temp[:-pos]
            # Update position and skip size
            pos = (pos + len_ + skip_size) % n
            skip_size += 1

    return vals


def densify(vals: list) -> str:
    """Makes a dense hex representation of a knot hash."""
    hex_parts = []
    block_size = 16

    for i in range(0, len(vals), block_size):
        # XOR each block together (0 ^ n always gives n)
        running = 0
        for val in vals[i: i+block_size]:
            running = running ^ val

        # Determine the two/digit hex representation of the block
        hex_ = hex(running)[2:]
        if len(hex_) == 1:
            hex_ = "0"+hex_
        hex_parts.append(hex_)

    res = "".join(hex_parts)
    return res


def solve(data: str, n=256):
    lengths = parse(data)
    vals = emulate_knot(lengths, n=n)
    star1 = vals[0] * vals[1]
    print(f"Solution to part 1: {star1}")

    lengths2 = parse(data, as_ascii_codes=True)
    vals_repeat = emulate_knot(lengths2, n=n, n_repititions=64)
    star2 = densify(vals_repeat)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 10
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
