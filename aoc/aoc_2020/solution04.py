# . .··`·* ` ·• .`  ·*·. `   · · + `.* ··   +`.·   ·.    ·. `·*   ·  .·  · .  `·
# `.   ·•.*··     *. ·`  · ·.  Passport Processing .     `·  *.·   •·` .  ·   ·.
# .· *·. ·`   .  ··    https://adventofcode.com/2020/day/4 · `     ·.      `·.+·
# ··*·.      `·· *.   .·   .·`  ·      `·*   . `. · *  ·       `·*..`·  ·  ·.*·`

import re
from typing import Callable

required_fields = {"byr", "iyr", "eyr", "hgt", "hcl", "ecl", "pid"}
optional_fields = set(["cid"])


def parse(s: str) -> list[dict[str, str]]:
    entries = s.split("\n\n")
    res = []
    for entry in entries:
        d = {}
        for part in entry.split():
            field, value = part.split(":")
            d[field] = value
            #
        res.append(d)
    return res


def _validate_height(s):
    """Separate function for height validation (could define that as a simple lambda)"""
    if not re.match(r"\d{2,3}\w{2}", s):
        return False
    val = int(s[:-2])
    unit = s[-2:]
    if unit == "cm":
        return 150 <= val <= 193
    elif unit == "in":
        return 59 <= val <= 76
    else:
        return False
    #


rules = {
    "byr": lambda s: bool(re.match(r"\d{4}", s)) and 1920 <= int(s) <= 2002,
    "iyr": lambda s: bool(re.match(r"\d{4}", s)) and 2010 <= int(s) <= 2020,
    "eyr": lambda s: bool(re.match(r"\d{4}", s)) and 2020 <= int(s) <= 2030,
    "hgt": _validate_height,
    "hcl": lambda s: bool(re.match(r"#[0-9a-f]{6}", s)),
    "ecl": lambda s: s in "amb blu brn gry grn hzl oth".split(),
    "pid": lambda s: bool(re.match(r"^\d{9}$", s))
}


def isvalid(entry, rules: dict[str, Callable[[str], bool]]|None=None) -> bool:
    """Determines whether a passport is valid.
    rules is an optional dict mapping fields to validation functions, all of which must
    return True for succesful validation."""

    if not all(field in entry for field in required_fields):
        return False
    if rules:
        return all(validator(entry[field]) for field, validator in rules.items())        
    else:
        return True
    #


def solve(data: str) -> tuple[int|str, ...]:
    parsed = parse(data)

    star1 = sum(isvalid(entry) for entry in parsed)
    print(f"Solution to part 1: {star1}")

    star2 = sum(isvalid(entry, rules) for entry in parsed)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2020, 4
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
