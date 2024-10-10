import re
import string


def parse(s):
    res = s
    return res


def decompress(s):
    """Decompresses a string"""
    letters = set(string.ascii_letters)
    res = ""
    ind = 0
    while ind < len(s):
        # If encountering a letter, just use the letter and move to the next character
        if s[ind] in letters:
            res += s[ind]
            ind += 1
        # If encountering a decompression marker, we need to parse it and expand the subsequent substring
        else:
            # Sanity check and match the decompression marker pattern
            if s[ind] != "(":
                raise ValueError("This should not happen")
            m = re.match(r"\((\d+)x(\d+)\)", s[ind:])
            if m is None:
                raise ValueError(f"Couldn't identify decompressiong marker: {s[ind:ind+20]}.")

            # Parse the marker
            marker_length = len(m.group(0))
            n_chars_repeat = int(m.group(1))
            n_times_repeat = int(m.group(2))

            # Expand subsequent substring and move pointer.
            ind += marker_length
            decompressed_substring = n_times_repeat*s[ind:ind+n_chars_repeat]
            res += decompressed_substring
            ind += n_chars_repeat

    return res


def compute_recursively_decompressed_string_length(s, running=0):
    res = running
    if not s:
        return res

    char = s[0]

    if char in string.ascii_letters:
        res += 1
        remainder = s[1:]
    elif char == "(":
        m = re.match(r"\((\d+)x(\d+)\)", s)
        marker_length = len(m.group(0))
        n_chars_repeat = int(m.group(1))
        n_times_repeat = int(m.group(2))

        # Recursively decompress the substring affected by the decompression marker
        snippet_affected_by_marker = s[marker_length:marker_length+n_chars_repeat]
        decomp = compute_recursively_decompressed_string_length(snippet_affected_by_marker)

        res += n_times_repeat*decomp

        # Discard the marker and decompressed substring and proceed
        remainder = s[marker_length+n_chars_repeat:]
    else:
        raise ValueError

    return res + compute_recursively_decompressed_string_length(remainder)


def solve(data: str):
    parsed = parse(data)

    decompressed = decompress(parsed)
    star1 = len(decompressed)
    print(f"Solution to part 1: {star1}")

    star2 = compute_recursively_decompressed_string_length(parsed)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2016, 9
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
