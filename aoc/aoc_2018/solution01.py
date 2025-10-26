# ·  *. ·.  ·`.   ··  .*` + ·.    .·   ·*.  `  ·+.  ·`+·.   *`. ·    . ·` +.·*` 
# .·+ •  `·  · .· `    ·* ·*.  Chronal Calibration  .    ·    `.*     ·  ·.·* .`
# .``· . ·    *·  .    https://adventofcode.com/2018/day/1   ·*` ·.  · `*.·+.` ·
# * ·.·   `· .·`    ·   •`.· +      ·. ` ·  · ·*   .   · `.·.*·  ` · * .·  `.··+


def parse(s: str):
    res = [int(part) for line in s.splitlines() for part in line.split(", ")]
    return res


def first_repeated_frequency(numbers: list[int]) -> int:
    """Repeatedly increments frequency, starting from zero, with numbers from the list, looping
    over the numbers indefinitely, until a frequency is observed a second time."""
    running = 0
    seen = {running}
    i = 0
    while True:
        number = numbers[i]
        running += number
        
        if running in seen:
            return running
        
        seen.add(running)
        i = (i+1) % len(numbers)
    #


def solve(data: str) -> tuple[int|str, int|str]:
    numbers = parse(data)

    star1 = sum(numbers)
    print(f"Solution to part 1: {star1}")

    star2 = first_repeated_frequency(numbers)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2018, 1
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()