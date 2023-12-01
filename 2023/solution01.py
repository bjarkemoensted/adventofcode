def read_input():
    with open("input01.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = s.split("\n")
    return res


def get_first_and_last_digits(s):
    digits = set("0123456789")
    digits_in_string = [char for char in s if char in digits]
    res = int(digits_in_string[0] + digits_in_string[-1])
    return res


def match_digit_or_word(s, reverse=False):
    digits = set("0123456789")
    words = "one, two, three, four, five, six, seven, eight, nine".split(", ")
    digits_words = dict(zip(words, map(str, range(1, 10))))

    inds = list(range(len(s)))
    if reverse:
        inds = inds[::-1]

    for i in inds:
        if s[i] in digits:
            return s[i]
        for word, n in digits_words.items():
            if s[i:].startswith(word):
                return n


def get_first_and_last_numbers(s):
    first = match_digit_or_word(s)
    last = match_digit_or_word(s, reverse=True)
    res = int(first+last)
    return res


def main():
    raw = read_input()
    parsed = parse(raw)

    calibration_values = [get_first_and_last_digits(line) for line in parsed]
    star1 = sum(calibration_values)
    print(f"Sum of calibration values is: {star1}.")

    updated_cal_vals = [get_first_and_last_numbers(line) for line in parsed]
    star2 = sum(updated_cal_vals)
    print(f"Updated values: {star2}")


if __name__ == '__main__':
    main()
