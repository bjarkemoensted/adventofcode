# ·. ·`` .·*·  + · * .  `· +. `  ·•·   . `  ··    *`•` .·  *· `  .* ··`.    ··.•
#  · `*+.`    · ·. ` ·   +.  Squares With Three Sides ·     * ·`.  ·    ·.  *`·.
# `+· .·• `  *.·•` ··  https://adventofcode.com/2016/day/3   ·+   `.* · .` · * ·
# +·.·  ` *.·`    ·  `·*.· .     `·.    ·   *  · .`  ·   ·+ `. · `·•   ·+ ·.. `·


def parse(s: str):
    res = []
    for line in s.split("\n"):
        sides = [int(x) for x in line.split()]
        res.append(sides)

    return res


def triangle_is_possible(sides):
    ordered = sorted(sides)
    return sum(ordered[:-1]) > ordered[-1]


def solve(data: str) -> tuple[int|str, int|str]:
    parsed = parse(data)

    star1 = sum(triangle_is_possible(sides) for sides in parsed)
    print(f"Solution to part 1: {star1}")

    new_sides = []
    for ind in range(len(parsed[0])):
        these_sides = [col[ind] for col in parsed]
        for i in range(0, len(these_sides), 3):
            new_sides.append(these_sides[i:i+3])

    star2 = sum(triangle_is_possible(sides) for sides in new_sides)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2016, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()