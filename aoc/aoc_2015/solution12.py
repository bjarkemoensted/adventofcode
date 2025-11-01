# `·. · +·* .` •      +. *·  ·* ` ·.  + ·   ·`  +* .··    *   ·+.·`  .   * ·. `·
# ·` .+*·.     · ·*.*    ·•  . JSAbacusFramework.io . ·  *`·.   ·    ·.*   .·`· 
# .··* `.     ·*·. ·`  https://adventofcode.com/2015/day/12 ··•*  +·  ·  ·  ·*.`
#  + ·.·* ·`·*`     ·   .+` · ·*.` ·      •.*·  `·.·    + ·   .·*`   ` ·*. ` ·+·


import json


def parse(s: str):
    return json.loads(s)


def recursive_iterate(obj):
    if isinstance(obj, int):
        yield obj
    elif isinstance(obj, list):
        for elem in recursive_iterate(obj):
            yield elem
        #
    elif isinstance(obj, dict):
        for elem in recursive_iterate(list(obj.values())):
            yield elem


def grab_numbers(obj, res=None, ignore_red=False):
    if res is None:
        res = []

    if isinstance(obj, int):
        res.append(obj)
    elif isinstance(obj, dict):
        if not (ignore_red and 'red' in obj.values()):
            for k, v in obj.items():
                grab_numbers(v, res=res, ignore_red=ignore_red)
    elif isinstance(obj, list):
        for elem in obj:
            grab_numbers(elem, res=res, ignore_red=ignore_red)
    return res


def solve(data: str) -> tuple[int|str, int|str]:
    parsed = parse(data)

    numbers = grab_numbers(parsed)

    star1 = sum(numbers)
    print(f"Solution to part 1: {star1}")

    new_numbers = grab_numbers(parsed, ignore_red=True)

    star2 = sum(new_numbers)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 12
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
