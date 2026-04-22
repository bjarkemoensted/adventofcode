# ` `·*.·  ·      +·  * . `··   *·+  * `.· *   .*  ·+·    `+ ·.*    . ·*+.· `*·`
#  .·* ·.`     .    +··    .* ·  Secure Container · `.   · ·+ *     ·   .  `*· .
#  ·*` ·  . ·  + *  ·  https://adventofcode.com/2019/day/4  ·      · ** ·     · 
# ·*..·   `    · .·     ·`*•·.      .` * +·   `·  . ·   .*· `   ·· .  ` * •·. *·



def parse(s: str) -> tuple[int, int]:
    a, b = map(int, s.split("-"))
    return a, b


def has_adjacent(s: str, n=2, exact=False) -> bool:
    """Whether the password has n repeated characters.
    if exact is True, groups with >= n repetitions do not count"""

    current = ""
    count = 0
    for i, char in enumerate(s):
        # Reset current character when encountering a new one
        if char != current:
            current = char
            count = 0
        
        # Check if this position meets the criterion
        count += 1
        if count == n:
            # Current group ends character is final, or next one is different
            group_ends_now = (i + 1 >= len(s) or s[i+1] != char)
            if group_ends_now or not exact:
                return True
            #
        #
    return False


def is_isotonic(pw: str) -> bool:
    """Whether the password is non-decreasing"""
    return all(pw[i+1] >= pw[i] for i in range(len(pw)-1))


def solve(data: str) -> tuple[int|str, ...]:
    low, high = parse(data)

    candidates = [
        pw for pw in map(str, range(low, high+1))
        if is_isotonic(pw) and has_adjacent(pw)
    ]
    
    star1 = len(candidates)
    print(f"Solution to part 1: {star1}")

    no_groups = [pw for pw in candidates if has_adjacent(pw, exact=True)]
    star2 = len(no_groups)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2019, 4
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
