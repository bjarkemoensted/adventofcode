# ` ⸳  ⸳+.* ⸳` ꞏ      `.ꞏ•⸳    ꞏ ꞏ`   ⸳ ꞏ.`  ꞏ  *⸳`⸳  .* ꞏ •. ꞏ.`* ⸳     *ꞏ•⸳.ꞏ.
# ꞏ.*`ꞏ  ` ⸳• + .ꞏ    *⸳ .  ꞏ    Inverse Captcha • .⸳   *  `+    ⸳ .*⸳ ꞏ   ⸳*`•ꞏ
# .ꞏ +  `*⸳ꞏ.  * ⸳. ꞏ  https://adventofcode.com/2017/day/1  `+⸳ ꞏ   .*  ꞏ⸳*  • ꞏ
# ⸳ ꞏ•`.. ⸳   ⸳   ⸳ +`⸳  ⸳ .ꞏ+       .ꞏ    ⸳+.ꞏ`    .⸳ ꞏ* .  ⸳ ⸳ .ꞏ         ꞏ⸳`*


def parse(s):
    res = [int(char) for char in s]
    return res


def match_next(arr):
    elems = [n for i, n in enumerate(arr) if arr[(i+1) % len(arr)] == n]
    return elems


def match_halfway(arr):
    elems = [n for i, n in enumerate(arr) if arr[(i+len(arr)//2) % len(arr)] == n]
    return elems


def solve(data: str):
    parsed = parse(data)

    star1 = sum(match_next(parsed))
    print(f"Solution to part 1: {star1}")

    star2 = sum(match_halfway(parsed))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 1
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()