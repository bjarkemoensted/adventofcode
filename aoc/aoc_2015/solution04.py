# · *`·`.·  · *+ ·  ·.   `+  ·.  +··*` · .·*  · · + ·    .`. * ··  *.+` ·    ·.·
# ` ·  ·     · · .    ·*`·  The Ideal Stocking Stuffer ·**  ·  `   .·*· * ··` ·.
# ·.··`       . ·*·  ` https://adventofcode.com/2015/day/4 ·*. * · `+·· .·* ·``·
# .·`*· *. ·  ·.` .+ ·  ·•  ` *··  ·   *· `.·  +   ·*    ··*  ·  .· *.  ·` + .·`


from hashlib import md5


def parse(s: str):
    res = s  # TODO parse input here
    return res


def make_hash(x):
    str_ = x.encode('utf8')
    hash_ = md5(str_).hexdigest()
    return hash_


def find_first_hash(startstring, data: str):
    """Finds the lowest positive integer n such that the MD5 hash of puzzle input + n starts with the startstring."""
    n = 0
    while not make_hash(data+str(n)).startswith(startstring):
        n += 1
        if n % 10000 == 0:
            print(n, end="\r")
    print()
    return n


def solve(data: str) -> tuple[int|str, int|str]:
    star1 = find_first_hash(5 * "0", data)
    print(f"Solution to part 1: {star1}")

    star2 = star2 = find_first_hash(6 * "0", data)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 4
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
