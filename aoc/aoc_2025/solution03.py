# *·.   `·+    ·.·+   ·  +  •·  ·`.·*  . ·*` +·    . ·  • · +    `•·· .* `·  .·`
# ·.·. · *  · •    ·`·    *.·`     *· Lobby  `*·  ·   ·• *.*·     ·.·*`  ·+  * ·
# .·`* *·.   ·  ·.  +  https://adventofcode.com/2025/day/3 .    ··.`* * ` . ·+·.
#  •*·.   ·+` ·.  *`·. +·.   · `*       .` ··. +`.  * `·   ` · .`* •  · ·.• . `·


def parse(s: str) -> list[list[int]]:
    res = [[int(char) for char in line] for line in s.splitlines()]
    return res


def max_joltage(array: list[int], n: int, running=0) -> int:
    """Determines the maximum joltage of n elements of an array of batteries.
    running is a running sum for recursing to the n-1 case."""

    # Base case when no batteries are left to add
    if n == 0:
        return running
    
    # Find index of the first occurrence of the greatest joltage. Leaves at least n-1 batteries
    cut = len(array) - n + 1
    ind, digit = max(enumerate(array[:cut]), key = lambda t: (t[1], -t[0]))
    
    # Add contribution of this battery to the sum. Recurse on remaining n-1 batteries
    contribution = digit*10**(n-1)

    return max_joltage(array[ind+1:], n-1, running + contribution)



def solve(data: str) -> tuple[int|str, ...]:
    batteries = parse(data)

    star1 = sum(max_joltage(array, n=2) for array in batteries)
    print(f"Solution to part 1: {star1}")

    star2 = sum(max_joltage(array, n=12) for array in batteries)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2025, 3
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
