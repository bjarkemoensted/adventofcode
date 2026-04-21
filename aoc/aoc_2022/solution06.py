# ·* ·. `·`    · `·+·.     `·+· ·+`  .· ·*    · `·+ .• ··   .·  `· +· .`· . *`··
# +·` ·   ·*. · · .  ·` · *.`· ·  Tuning Trouble *. · .`   +·     `·•  . ·* .·`•
# .·.` `*·  ··.`+ ·. * https://adventofcode.com/2022/day/6 .*·`·.+   · *  ·.· .`
# ·.  ··+`.· ` ·. *·   · `·+ .* `   ··* ·.`  ·+  ··.`  * .··` · *.·`   ··.`·  +.


def parse(s: str) -> str:
    return s.strip()


def find_package_start_location(s: str, marker_size=4) -> int:
    for i in range(len(s) - marker_size):
        snippet = s[i:i+marker_size]
        if len(set(snippet)) == marker_size:
            return i + marker_size
        #
    raise RuntimeError


def solve(data: str) -> tuple[int|str, ...]:
    parsed = parse(data)

    star1 = find_package_start_location(parsed)
    print(f"Solution to part 1: {star1}")

    star2 = find_package_start_location(parsed, marker_size=14)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 6
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
