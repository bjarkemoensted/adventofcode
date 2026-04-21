# · . *` · ·     ·.· *`  ·  *· .`+·  ·     .  *+`· *· .   ·   ·*.  ·` .  ·  *··`
# .`·* ·    ·`. *    ·    ·  Rucksack Reorganization *·  .   ·   •   ·* `.  *· ·
#  ·` ·+* ·  .` · ·.   https://adventofcode.com/2022/day/3 ·.•      ·•·  *.·  `·
# ·  .  ·` *.· ·`  *. ·`·*.·    · `   · ·+.`·       ·`* . · ·   *`·    ·  * ·`.+

import string


def parse(s: str) -> list[str]:
    return s.splitlines()


# Map each letter to a priority
letters = string.ascii_lowercase + string.ascii_uppercase
priority_map = {letter: i+1 for i, letter in enumerate(letters)}


def common_priority(*groups: str) -> int:
    """Finds the shared item in an iterable of strings representing items,
    and returns its priority"""
    
    # Find shared item among the groups
    overlap = set.intersection(*map(set, groups))
    assert len(overlap) == 1
    # Return priority
    item = overlap.pop()
    res = priority_map[item]

    return res


def rucksack_priorities(*rucksacks: str) -> int:
    """Sum the priorities of items in both compartments of the rucksacks"""
    res = 0
    for s in rucksacks:
        # Split into the compartments
        cut = len(s) // 2
        assert 2*cut == len(s)
        # Add priority
        res += common_priority(s[:cut], s[cut:])
    
    return res


def group_priorities(*rucksacks: str, groupsize=3) -> int:
    """Sum the priorities of items shared among each group"""
    res = 0

    for i in range(0, len(rucksacks), groupsize):
        group = rucksacks[i:i+groupsize]
        assert len(group) == groupsize
        res += common_priority(*group)
    
    return res


def solve(data: str) -> tuple[int|str, ...]:
    rucksacks = parse(data)

    star1 = rucksack_priorities(*rucksacks)
    print(f"Solution to part 1: {star1}")

    star2 = group_priorities(*rucksacks)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2022, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
