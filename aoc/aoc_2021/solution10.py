#  ·+.· ` ·*  ·   .*    `· *.  ` ·.  ·.  +·   ·*·.`  ·    .· · *`  ·. +` · .·  .
# · `.*    ··. `*.  + ·   ·.• ·   Syntax Scoring    ·. ·`+*.` .  ·    `· `.+*· ·
# ·`.·    +   .·  *.·  https://adventofcode.com/2021/day/10 ·.  *       ·•·*. · 
# .*·` ·  `  ·*.  ·`   * .  `·• .  *·   ·.` *· . ·+ ` . ·   . * ·.  `*·   *· . ·


def parse(s: str) -> list[str]:
    return s.splitlines()


SYNTAX_ERROR_POINTS = {
        ")": 3,
        "]": 57,
        "}": 1197,
        ">": 25137
    }


# Map bracket symbols to matching symbols
brackets = {'(': ')', '[': ']', '{': '}', '<': '>'}
brackets_inv = {v: k for k, v in brackets.items()}
# Store all the bracket symbols in a set
allowed = set.union(*map(set, brackets.items()))


def find_breaking_char(s: str) -> str|None:
    """Attempts to find breaking character. Returns it if successful, otherwise returns None"""
    chars = []
    for char in s:
        assert char in allowed
        # We can always open another parenthesis
        if char in brackets:
            chars.append(char)
            continue
        # Can only close the last parenthesis to be opened
        last = chars.pop()
        if last == brackets_inv[char]:
            # If last + current chars are e.g. '(' and ')', they cancel out
            pass
        else:
            return char
        #
    return None


def autocomplete(s: str) -> str:
    """Autocompletes the input string"""
    chars = []
    for char in s:
        assert char in allowed
        # We can always open another parenthesis
        if char in brackets:
            chars.append(char)
            continue
        # Can only close the last parenthesis to be opened
        last = chars.pop()
        if last == brackets_inv[char]:
            # If last + current chars are e.g. '(' and ')', they cancel out
            pass
        else:
            raise ValueError("This should not happen if the string is merely incomplete")
        #

    missing_brackets = []
    while chars:
        missing_brackets.append(brackets[chars.pop()])
    return "".join(missing_brackets)


def score_autocomplete_string(s: str) -> int:
    """Computes the autocomplete score for a string"""
    res = 0
    points = {c: i + 1 for i, c in enumerate(")]}>")}
    for char in s:
        res *= 5
        res += points[char]
    return res


def compute_syntax_error_score(line: str) -> int:
    """Compute the syntax error score for a line"""
    breaking = find_breaking_char(line)
    if breaking is None:
        return 0
    else:
        return SYNTAX_ERROR_POINTS[breaking]
    #


def compute_median_autocomplete_score(*incomplete_lines: str) -> int:
    """Compute the median autocomplete score for the input lines"""

    scores = []
    for line in incomplete_lines:
        missing_brackets = autocomplete(line)
        autocomplete_score = score_autocomplete_string(missing_brackets)
        scores.append(autocomplete_score)

    scores.sort()
    middle_ind = (len(scores) - 1) // 2
    res = scores[middle_ind]
    return res


def solve(data: str) -> tuple[int|str, ...]:
    lines = parse(data)

    error_scores = [compute_syntax_error_score(line) for line in lines]
    star1 = sum(error_scores)
    print(f"Solution to part 1: {star1}")

    incomplete_lines = [line for line, score in zip(lines, error_scores, strict=True) if score == 0]
    star2 = compute_median_autocomplete_score(*incomplete_lines)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2021, 10
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
