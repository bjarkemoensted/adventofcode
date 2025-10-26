# .*`· . ·+·.   .·`` · .•  ·`    *·   .` · ·`.* + ·· `·   `  .··.` ·*`  .·` .·*`
# ·`.  ·`   ·. ·`*.   · .·*`· Elves Look, Elves Say  *  `· ·.`  *  .  ` · • ` ·.
# `··`   ·.*`·+  ·  `• https://adventofcode.com/2015/day/10 ·  .`·*   ·   ·.·`  
# ·   ` · ·.  .`•  ·· `    ·* `·  .`*   ·  +·`·     .·`·+ ·`*    .· `  ·* .·` .·


def extend_sequence(s, n_times=1):
    """Extends a sequence, e.g. transforms "11" into "21" (two ones)."""

    if n_times == 0:
        return s

    res = ""
    buffer = ""

    for char in s:
        if buffer and char != buffer[-1]:
            res += str(len(buffer))+buffer[-1]
            buffer = ""
        buffer += char

    if buffer:
        res += str(len(buffer)) + buffer[-1]

    return extend_sequence(res, n_times - 1)


def solve(data: str) -> tuple[int|str, int|str]:
    extended = data
    for _ in range(40):
        extended = extend_sequence(extended)

    star1 = len(extended)
    print(f"Solution to part 1: {star1}")

    for _ in range(10):
        extended = extend_sequence(extended)
    star2 = len(extended)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 10
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()