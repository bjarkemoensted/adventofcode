{{ascii_header|safe}}


def parse(s):
    res = s  # TODO parse input here
    return res


def solve(data: str):
    parsed = parse(data)

    # TODO solve puzzle
    star1 = None
    print(f"Solution to part 1: {star1}")

    star2 = None
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = {{year}}, {{day}}
    {% if has_examples -%}
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    {% endif -%}
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
