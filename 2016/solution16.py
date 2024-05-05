def read_input():
    with open("input16.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = s
    return res


def dragon_curve_thingy(s: str) -> str:
    flip = {
        "0": "1",
        "1": "0"
    }

    s2 = "".join([flip[char] for char in s[::-1]])
    res = s + "0" + s2
    return res


def generate_data(initial_state: str, n_bits: int) -> str:
    s = initial_state
    while len(s) < n_bits:
        s = dragon_curve_thingy(s)

    res = s[:n_bits]
    return res


def _get_pairs(s: str) -> list:
    pairs = [s[i:i+2] for i in range(0, len(s), 2)]
    assert len(s) == sum(map(len, pairs))
    return pairs


def checksum(s: str) -> str:
    res = s
    while True:
        pairs = _get_pairs(res)
        res = "".join([str(int(a == b)) for a, b in pairs])

        if len(res) % 2 != 0:
            break
        #
    return res


def main():
    raw = read_input()
    initial_state = parse(raw)
    n_bits = 272

    data = generate_data(initial_state, n_bits)
    star1 = checksum(data)

    print(f"The checksum of the random-ish data is {star1}.")

    n_bits2 = 35651584
    data2 = generate_data(initial_state, n_bits2)
    star2 = checksum(data2)
    print(f"The checksum of the larger random-ish data is {star2}.")


if __name__ == '__main__':
    main()
