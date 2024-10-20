# `ꞏ      ⸳ `ꞏ⸳  .* ꞏ. .ꞏ⸳ *ꞏ⸳.⸳•   . `         ꞏ*.⸳  `⸳ ꞏ .ꞏ⸳`•   * ⸳ `•⸳ . ꞏ⸳`
#    ꞏ*  .•  .ꞏ    *⸳⸳ `   ꞏ      Not Quite Lisp  ⸳*.ꞏ   *⸳  ꞏ `    ⸳.* ꞏ. `• `⸳
# .* ⸳  .` ꞏ•          https://adventofcode.com/2015/day/1  .       `•` ⸳ ⸳ꞏ .  
#  ⸳` ꞏ   .   . +ꞏꞏ⸳*         ⸳ ⸳ꞏ.+   ⸳⸳    .ꞏ .`   *⸳    `  ꞏꞏ ⸳  * ꞏ⸳  *•   ꞏ


def solve(data: str):
    s = data

    d = {"(": 1, ")": -1}
    star1 = sum(d[char] for char in s)
    print(f"Solution to part 1: {star1}")

    running = 0
    for i, char in enumerate(s):
        running += d[char]
        if running < 0:
            print(i + 1)
            break
    star2 = i + 1
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 1
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
