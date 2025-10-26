# `·•·`*.  · · •  ·`+*·    *`·.  ·    . ·`+      ` .·   ` · *  .`·   +. ·  .`··*
# ·` . `· +  *·    ·  .`      ·` Packet Scanners ·   ` . ·+`   ·.*  • ·``.· ·.*`
#  .·   `·.*    ··  `  https://adventofcode.com/2017/day/13 .·   ·   · . ·` +*`·
# .* `·  ·`+.`* · * ·`    *.·  +`     · * ` ·` ·   *  `·.  +· ` ·.*  .·  `*·  ·.


import numba


def parse(s: str):
    res = []
    for line in s.splitlines():
        depth, range_ = [int(elem) for elem in line.strip().split(": ")]
        period = 2*range_ - 2
        scanner = (depth, range_, period)
        res.append(scanner)
    return res


@numba.njit
def severity(firewall):
    res = 0
    for depth, range_, period in firewall:
        caught = depth % period == 0
        sev = depth*range_
        if caught:
            res += sev

    return res


@numba.njit
def determine_delay(firewall):
    delay = 0
    while True:
        for depth, _, period in firewall:
            if (depth + delay) % period == 0:
                break
            #
        else:
            return delay
        delay += 1


def solve(data: str) -> tuple[int|str, int|str]:
    firewall = parse(data)

    star1 = severity(firewall)
    print(f"Solution to part 1: {star1}")

    star2 = determine_delay(firewall)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2017, 13
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()