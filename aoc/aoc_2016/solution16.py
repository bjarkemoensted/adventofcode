import numpy as np


char2bool = {"0": False, "1": True}
bool2char = {v: k for k, v in char2bool.items()}


def parse(s):
    arr = np.array([char2bool[char] for char in s])
    return arr


def dragon_curve_thingy(s: str) -> str:
    flip = {
        "0": "1",
        "1": "0"
    }

    s2 = "".join([flip[char] for char in s[::-1]])
    res = s + "0" + s2
    return res


def generate_data(initial_state: str, n_bits: int) -> str:
    a = initial_state.copy()
    while len(a) < n_bits:
        b = np.logical_not(np.flip(a.copy()))
        a = np.hstack([a, False, b])
        
    res = a[:n_bits]
    return res


def _get_pairs(s: str) -> list:
    pairs = [s[i:i+2] for i in range(0, len(s), 2)]
    assert len(s) == sum(map(len, pairs))
    return pairs


def checksum(a: np.ndarray) -> str:
    a = a.copy()
    while len(a) % 2 == 0:
        pairs = np.reshape(a, (-1, 2))
        a = np.logical_not(np.logical_xor.reduce(pairs, axis=1))
        
    res = "".join([bool2char[bool_] for bool_ in a])

    return res


def solve(data: str):
    example_disk_sizes = {
        "110010110100": 12,
        "10000": 20,
    }

    is_example = data in example_disk_sizes
    n_bits = example_disk_sizes[data] if is_example else 272
    initial_state = parse(data)

    data = generate_data(initial_state, n_bits)
    star1 = checksum(data)

    print(f"The checksum of the random-ish data is {star1}.")

    n_bits2 = 35651584
    if is_example:
        star2 = None
    else:
        data2 = generate_data(initial_state, n_bits2)
        star2 = checksum(data2)
        print(f"The checksum of the larger random-ish data is {star2}.")

    return star1, star2


def main():
    year, day = 2016, 16
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve, extra_kwargs_parser="ignore")
    from aocd import get_data
    raw = get_data(year=year, day=day)
    
    solve(raw)


if __name__ == '__main__':
    main()
