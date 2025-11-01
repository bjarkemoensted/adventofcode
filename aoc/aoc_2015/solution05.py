# · ..*` ·  · *.  ·` . * · `*.•·  · `   + ·*.  ··`.`·  * ·      ·  .+` ·*  . ·`+
#  .·` *·.`  `· .*    Doesn't He Have Intern-Elves For This? · * •· . *.·`   *·.
# `*  ·   .·  + `·.· • https://adventofcode.com/2015/day/5 +* ··    `+. `.` · *`
# .·`  ·  *.`·  •  . `  .`·*     ·`+.   · *`·+·.. `*   .·•`·  +  ·.` .·  *· `. ·


def parse(s: str):
    res = [line for line in s.split("\n")]
    return res


def string_is_nice(s):
    """Determines if a string is 'nice', as specified by Santa."""
    vowels = 'aeiou'
    forbidden = ['ab', 'cd', 'pq', 'xy']

    # Must have 3 vowels
    if sum(char in vowels for char in s) < 3:
        return False
    # Must have repeating letters
    if not any(s[i] == s[i+1] for i in range(len(s)-1)):
        return False
    # Must not contain naughty substrings
    if any(nope in s for nope in forbidden):
        return False

    return True


def string_is_nice2(s):
    if not any(s[i:i+2] in s[i+2:] for i in range(len(s) - 2)):
        return False
    if not any(s[i] == s[i+2] for i in range(len(s) - 2)):
        return False
    return True


def solve(data: str) -> tuple[int|str, int|str]:
    strings = parse(data)

    star1 = sum(string_is_nice(s) for s in strings)
    print(f"Solution to part 1: {star1}")

    star2 = sum(string_is_nice2(s) for s in strings)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 5
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
