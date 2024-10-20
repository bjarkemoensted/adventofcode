# ꞏ •   .  ` + `.⸳ ꞏ.•     .ꞏ*ꞏ  `⸳   * ꞏ      * .⸳  .`   ꞏ*  ⸳      ` •ꞏ ⸳*`.  
# `⸳ `  +  ꞏ  ⸳ * ꞏ ꞏ+   . ⸳*` ꞏ  ꞏ  Spinlock  ⸳`*   +ꞏ     ꞏ ` . • ꞏꞏ `+    • ꞏ
#  ` * • ꞏ⸳.*      * . https://adventofcode.com/2017/day/17 . +   ꞏ `*  *.`ꞏ ⸳ *
# *•⸳  ꞏ `   .    + ⸳ꞏ`.⸳  *   ` ꞏ• .. *   `⸳*ꞏ    ⸳•ꞏ  ` * +ꞏ* `.      ꞏ⸳+ ꞏ ⸳.


import numba


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
    year, day = 2017, 17
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
