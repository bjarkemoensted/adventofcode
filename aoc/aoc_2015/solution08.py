# ·.` •    `` .    *`.· ·.  *` ·*    `+  ·  . ·` *.   ·``•·    ·  ` ·*. ·• .` *·
#  •+`· .·.  ` · •.   `·*  .· +    Matchsticks ·.    `    .`· ·      `    ·•.+·`
# *`··.  ` · · .*  `   https://adventofcode.com/2015/day/8 *`•   ·   ·   `.  ·`•
# .·• `   ·  + *.·`       `·.*  ` · .• `.* ·. `*  ·  ·  + `     .•·`   ·`. ·*` .


def parse(s: str):
    res = [line.strip() for line in s.split("\n")]
    return res


def get_code_length(s):
    return len(eval(s))


def get_string_length(s):
    return len(s)


def get_encoded_string_length(s):
    len_ = 0

    n_doubleslash = s.count('\\')
    len_ += 2 * n_doubleslash
    s = s.replace('\\', '')

    n_escape_quotes = s.count('"')
    len_ += 2 * n_escape_quotes
    s = s.replace('"', '')

    x_escape_inds = [i for i in range(len(s) - 2) if s[i:i + 2] == '\\x']
    escapes = [s[i: i + 4] for i in x_escape_inds]
    len_ += 5 * len(escapes)
    for esc in escapes:
        s = s.replace(esc, '')

    len_ += len(s) + 2
    return len_


def solve(data: str) -> tuple[int|str, int|str]:
    lines = parse(data)

    total_string_length = sum(get_code_length(line) for line in lines)
    total_code_length = sum(get_string_length(line) for line in lines)

    star1 = total_code_length - total_string_length
    print(f"Solution to part 1: {star1}")

    total_encoded_length = sum(get_encoded_string_length(line) for line in lines)
    star2 = total_encoded_length - total_code_length
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 8
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()