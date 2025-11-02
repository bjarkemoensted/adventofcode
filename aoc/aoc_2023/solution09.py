# *. ·` · •·`    *· .·   *`.·   *   .· ·`    ·.*··` · .+ *   ·    `· +  ·* .·` ·
# ·`   ·*`· .  * ·•·` . ·.· •   Mirage Maintenance · .•·  *`  · `. +·  · `.*· .·
# ` ·.*  ·   ·` · .    https://adventofcode.com/2023/day/9  ·.`·· *  ·` •·· *.·`
#  ·.*·`  ·  *.·     `*  ·.   `+ ·  ·*·`•·    ` · *. ·` +.·    .  ·`.· `·  · •` 


def parse(s: str) -> list[list[int]]:
    res = [[int(elem) for elem in line.split()] for line in s.split("\n")]
    return res


def extrapolate(numbers) -> int:
    if all(n == 0 for n in numbers):
        return 0

    diffs = [numbers[i] - numbers[i-1] for i in range(1, len(numbers))]
    return numbers[-1] + extrapolate(diffs)


def solve(data: str) -> tuple[int|str, ...]:
    parsed = parse(data)

    star1 = sum(extrapolate(numbers) for numbers in parsed)
    print(f"Solution to part 1: {star1}")

    star2 = sum(extrapolate(numbers[::-1]) for numbers in parsed)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2023, 9
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
