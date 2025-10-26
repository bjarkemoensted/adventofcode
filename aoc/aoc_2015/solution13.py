# .`• ·*. ·   * `·`*.· . ·*`+.  ·`   . +   . ·· `   *·+.  `· *· `· ·.  •  · ·*.`
# ·.`.*·•`  .  · *·  * `   Knights of the Dinner Table  • ·.· • `*.   •. ·`.*··.
# *·· .`·*  · . ·`+.   https://adventofcode.com/2015/day/13      .·*  . ·  `.  ·
# ·..*   ·  +`     · `.··     .· · • ` .· .·` *· .   *··. `   . *`•   ·· . ··` .


from itertools import permutations


def parse(s: str) -> dict[str, dict[str, int]]:
    """Parses into e.g. {'Alice': {"Bob": -42, ...}, ...}"""

    res: dict[str, dict[str, int]] = {}
    for line in s.split("\n"):
        words = line.split(" ")
        a = words[0]
        b = words[-1].replace('.', '')
        score = int(words[3]) * (-1 if words[2] == "lose" else 1)
        if a in res:
            res[a][b] = score
        else:
            res[a] = {b: score}

    return res


def compute_happiness(arrangement, prefdict):
    score = 0
    for i, name in enumerate(arrangement):
        neighbors = [arrangement[(i+inc) % len(arrangement)] for inc in (-1, 1)]
        person_prefs = prefdict.get(name, {})
        contribution = sum(person_prefs.get(neighbor, 0) for neighbor in neighbors)
        score += contribution

    return score


def determine_optimum_arrangement(names, preferences):
    res = float('-inf')

    for arrangement in permutations(names):
        happiness = compute_happiness(arrangement, preferences)
        res = max(res, happiness)

    return res


def solve(data: str) -> tuple[int|str, int|str]:
    prefs = parse(data)

    names = sorted(prefs.keys())
    star1 = determine_optimum_arrangement(names=names, preferences=prefs)
    print(f"Solution to part 1: {star1}")

    names_with_me = [name for name in names] + ["Me"]
    star2 = determine_optimum_arrangement(names=names_with_me, preferences=prefs)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 13
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()