# .·  · *• · .·  ` *·.` +      +  ·    ·`. * ·   *.·  ` · .•·   *.·   ` + . `  ·
# *·`·   . *··  ` ·   *· `   ·+·* Not Quite Lisp  ` ·      . *` ·  `·*    *`.·· 
# · .* `·    ` *. ·    https://adventofcode.com/2015/day/1 ·` ·`  * .· ` • ··.`*
#  .•` ·`·   * ·  .` +  ·  *·.   ·`  ·+·*  .•·` ·  `*·    ·`*.  `• ·*    ·  *·.`


def solve(data: str) -> tuple[int|str, int|str]:
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


def main() -> None:
    year, day = 2015, 1
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
