# ·.*` ·.·` *·+ `.  *  ··   . ·` ·· .*`   . *· · `.· + . ··`. *   .`··   `*·  .·
# *·• `.·* ·   ·  `*.· ·   .  ·  Calorie Counting    ·*·.` +·•·.   ·• . `*· ·.·`
# ` ·.·*   `  · . ·` * https://adventofcode.com/2022/day/1 ·   `· ·*. *·· ` `·*.
# ·*.·.`+.  ·   ·*+· `.  ·. *`+   ·   .` ·  ·`. ·*·*    ·.` +·.·` `  *  ·· *.•·+


def parse(s: str) -> list[list[int]]:
    snippets = s.split("\n\n")
    calories = [[int(substring) for substring in snippet.splitlines()] for snippet in snippets]
    return calories


def solve(data: str) -> tuple[int|str, ...]:
    calories = parse(data)

    totals = [sum(arr) for arr in calories]
    totals.sort(reverse=True)
    star1 = totals[0]
    print(f"Solution to part 1: {star1}")

    star2 = sum(totals[:3])
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 1
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
