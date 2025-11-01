# ·+··.*  ·`*.·    ··*.`·  `*·     ·•  `·· +.   *  ` ·· `· ·  .·* ·`+·   `+.··•·
# ·*`  `·+`.·   ·*`  ·· `• ·    Plutonian Pebbles `·    ·`·*.    ··*`+ ·  ·· .`·
# *··+` . ·*`· ·` ·    https://adventofcode.com/2024/day/11 ·*  ·` ·. · *·.+` ·`
# .·  ·  ·*·   ` ·*  ·  .·· * ·.`·· ·*   . ·   ·  · .* · *·  `·+  . · ` ··*`· ·.


from collections import defaultdict
from functools import cache


def parse(s: str) -> dict[int, int]:
    """Parses into a dict where each number is mapped to the number of stones it appears on"""
    res: dict[int, int] = defaultdict(lambda: 0)
    for stone in map(int, s.strip().split()):
        res[stone] += 1
    
    return res


@cache
def update_single_stone(number: int):
    """Given a stone number, returns a list of the stone(s) which will result from blinking."""

    if number == 0:
        return [1]
    
    # If even number of digits, split into two (e.g. 1337 -> [13, 37])
    digits = str(number)
    if len(digits) % 2 == 0:
        cut = len(digits) // 2
        return [int(part) for part in (digits[:cut], digits[cut:])]
    
    return [number*2024]


def blink(stones: dict, n: int=1) -> dict[int, int]:
    """Blinks n times, updating all stones in each iteration.
    Stones is assumed to by a dictionary type, where each number is mapped to the number of times it occurs."""

    if n < 0:
        raise ValueError
    
    for _ in range(n):
        d_next: dict[int, int] = defaultdict(lambda: 0)
        for stone, multiplicity in stones.items():
            for val in update_single_stone(stone):
                d_next[val] += multiplicity
            #
        stones = d_next
    
    return stones
    

def solve(data: str) -> tuple[int|str, int|str]:
    stones = parse(data)
    
    n_steps = 25
    updated = blink(stones, n=n_steps)
    star1 = sum(updated.values())
    print(f"Solution to part 1: {star1}")

    n_steps_part2 = 75
    n_remaining = n_steps_part2 - n_steps
    updated = blink(updated, n=n_remaining)
    star2 = sum(updated.values())
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2024, 11
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
