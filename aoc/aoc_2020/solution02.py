# ·`.`* `·.· *   ·`. +   · . ·  `.·.*  · . `.*· ·*` + `·     *· . `        ` ·*·
# `.··  *  `·   ·  *.`•·   · ` Password Philosophy     .· +`·     · • . ` * .·.`
# · * ·. `·*  .`*.·    https://adventofcode.com/2020/day/2  .`    *· ·`   .*·``+
# .·` ` ·*    `* `.· ·. · `  *+. `   ·   ·  *` .*`·  . •`  ·  .· +   `·  ·`.`•·*

import re


def parse(s: str) -> list[tuple[int, int, str, str]]:
    pattern = r"(\d*?)\-(\d*?) (\w): (.*)"
    res: list[tuple[int, int, str, str]] = []

    for line in s.splitlines():
        matches = re.findall(pattern, line)
        assert len(matches) == 1
        m = matches[0]
        a, b = (int(s) for s in m[:2])
        let = m[2]
        pw = m[3]
        t = (a, b, let, pw)
        res.append(t)

    return res


def isvalid_old(n_min, n_max, letter, password) -> bool:
    """Checks password validity according to the old policy"""
    valid = n_min <= password.count(letter) <= n_max
    return valid

def isvalid_new(pos1, pos2, letter, password) -> bool:
    """Checks password validity according to the new policy"""
    valid = (password[pos1-1] == letter) ^ (password[pos2-1] == letter)
    return valid


def solve(data: str) -> tuple[int|str, ...]:
    elems = parse(data)

    star1 = sum((isvalid_old(*t) for t in elems))
    print(f"Solution to part 1: {star1}")

    star2 = sum((isvalid_new(*t) for t in elems))
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 2
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
