# `.·  ·   . ·` .  · * ··`· .+·  `* ·. ·•` ·+     *·.`  ··   ·`*   ·` *·.` · ·.*
# *• ·· ` ··  * ·` . · .`·   ·   Dragon Checksum .     · *.`· · `   * ·     ·*·`
# ··` .`·`     ·*·  ·` https://adventofcode.com/2016/day/16  ` ·*   .·` ·+· . `·
# · · +· .  ` ·*` ·   ·  . ·*. ·* · •`* .   ·` ·   *·.·`·  *    · . ·*   · ·`.*·

import numpy as np
from numpy.typing import NDArray

char2bool = {"0": False, "1": True}
bool2char = {v: k for k, v in char2bool.items()}


def parse(s: str) -> NDArray[np.bool_]:
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


def generate_data(initial_state: NDArray[np.bool_], n_bits: int) -> NDArray[np.bool_]:
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


def checksum(a: NDArray[np.bool_]) -> str:
    a = a.copy()
    while len(a) % 2 == 0:
        pairs = np.reshape(a, (-1, 2))
        a = np.logical_not(np.logical_xor.reduce(pairs, axis=1))
        
    res = "".join([bool2char[bool_] for bool_ in a])

    return res


def solve(data: str) -> tuple[int|str, int|str]:
    n_bits = 272
    initial_state = parse(data)

    generated_data = generate_data(initial_state, n_bits)
    star1 = checksum(generated_data)

    print(f"The checksum of the random-ish data is {star1}.")

    n_bits2 = 35651584
    data2 = generate_data(initial_state, n_bits2)
    star2 = checksum(data2)
    print(f"The checksum of the larger random-ish data is {star2}.")

    return star1, star2


def main() -> None:
    year, day = 2016, 16
    from aocd import get_data
    raw = get_data(year=year, day=day)
    
    solve(raw)


if __name__ == '__main__':
    main()
