import re
import string


def read_input():
    with open("input09.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


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


def main():
    raw = read_input()
    parsed = parse(raw)

    test_data = {
        "ADVENT": "ADVENT",
        "A(1x5)BC": "ABBBBBC",
        "(3x3)XYZ": "XYZXYZXYZ",
        "A(2x2)BCD(2x2)EFG": "ABCBCDEFEFG",
        "(6x1)(1x3)A": "(1x3)A",
        "X(8x2)(3x3)ABCY": "X(3x3)ABC(3x3)ABCY"
    }

    decompressed = decompress(parsed)
    print(f"The decompressed string contains {len(decompressed)} characters.")


if __name__ == '__main__':
    main()
