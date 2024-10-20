#  . *ꞏ  `    ⸳ ꞏ.`•⸳ `ꞏ⸳•.       ꞏ`.•  ⸳* .+ .ꞏ   .` ⸳* `ꞏ ꞏ +  *`⸳  .     `*⸳ꞏ
# ``   .  `ꞏ` ⸳•`* ꞏ..  I Was Told There Would Be No Math •`.ꞏ    ⸳  `+. ꞏ+⸳ꞏ` .
#  ꞏ`⸳*   . +     .`*  https://adventofcode.com/2015/day/2 .• ꞏ ⸳ꞏ. `• `   ꞏ . ⸳
# ꞏ⸳• `⸳.     ꞏ`*  .  ⸳* `ꞏ.` ⸳*  ꞏ.    +.⸳ꞏ.`•   ꞏ  ` +ꞏ⸳   +  ꞏ   .  •⸳ `  ꞏ.`


def parse(s):
    res = [tuple(int(elem) for elem in line.split("x")) for line in s.split("\n")]
    return res


def compute_wrapping_paper_area(dimensions):
    a, b, c = dimensions
    sides = [a*b, b*c, a*c]
    area = 2*sum(sides)
    slack = min(sides)
    return area + slack


def compute_ribbon_length(dimensions):
    a, b, c = dimensions
    volume = a*b*c
    shortest_edges = sorted(dimensions)[:2]
    circumference = 2*sum(shortest_edges)
    return circumference + volume


def solve(data: str):
    parsed = parse(data)

    star1 = sum([compute_wrapping_paper_area(dimension) for dimension in parsed])
    print(f"Solution to part 1: {star1}")

    star2 = sum(compute_ribbon_length(dimension) for dimension in parsed)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 2
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
