# ·`+ ··` .   *·`·+•  ·`  .··    +.*` ·`  .+· . *·     ·.`   `*+      · .*· `· `
#  ·*·   `*  .·.*  .·•`·     *·`  Custom Customs . · ` *  ·  +` ·   ·`  ·· `·.*+
# ·+`. `·· * ·   *   ` https://adventofcode.com/2020/day/6 ·.· ·`* · + ·* `.  ··
# ` ·  ·*       · .·   + · ` ·•.   · . ·`.  +·    `.·• `*· .+ ·`•`·      .*·+·.`

from collections import Counter


def parse(s: str) -> list[tuple[int, dict[str, int]]]:
    res = []
    for chunk in s.split("\n\n"):
        lines = chunk.splitlines()
        n_persons = len(lines)
        counts: dict[str, int] = Counter(sum([list(s) for s in lines], []))
        res.append((n_persons, counts))

    return res


def count_yes(forms: list[tuple[int, dict[str, int]]]) -> int:
    return sum((len(counts) for _, counts in forms))


def count_all_yes(forms: list[tuple[int, dict[str, int]]]) -> int:
    return sum(sum(v == n for v in counts.values()) for n, counts in forms)


def solve(data: str) -> tuple[int|str, ...]:
    forms = parse(data)

    star1 = count_yes(forms)
    print(f"Solution to part 1: {star1}")

    star2 = count_all_yes(forms)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 6
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
