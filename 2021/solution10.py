with open("input10.txt") as f:
    lines = [line.strip() for line in f]

brackets = ["()", "[]", "{}", "<>"]
open2close = {}
close2open = {}
allowed = set(sum((list(s) for s in brackets), []))
for open_, close_ in (list(s) for s in brackets):
    open2close[open_] = close_
    close2open[close_] = open_


def find_breaking_char(s):
    chars = []
    for char in s:
        assert char in allowed
        # We can always open another parenthesis
        if char in open2close:
            chars.append(char)
            continue
        # Can only close the last parenthesis to be opened
        last = chars.pop()
        if last == close2open[char]:
            # If last + current chars are e.g. '(' and ')', they cancel out
            pass
        else:
            return char
        #
    return None


syntax_error_score = 0
points = {
    ")": 3,
    "]": 57,
    "}": 1197,
    ">": 25137
}

incomplete_lines = []
for line in lines:
    breaking = find_breaking_char(line)
    if breaking is None:
        incomplete_lines.append(line)
    else:
        syntax_error_score += points[breaking]

print(f"Solution to star 1: {syntax_error_score}.")


def autocomplete(s):
    chars = []
    for char in s:
        assert char in allowed
        # We can always open another parenthesis
        if char in open2close:
            chars.append(char)
            continue
        # Can only close the last parenthesis to be opened
        last = chars.pop()
        if last == close2open[char]:
            # If last + current chars are e.g. '(' and ')', they cancel out
            pass
        else:
            raise ValueError("This should not happen if the string is merely incomplete")
        #

    missing_brackets = []
    while chars:
        missing_brackets.append(open2close[chars.pop()])
    return "".join(missing_brackets)


def score_autocomplete_string(s):
    res = 0
    points = {c: i + 1 for i, c in enumerate(")]}>")}
    for char in s:
        res *= 5
        res += points[char]
    return res


autocomplete_scores = []
for line in incomplete_lines:
    missing_brackets = autocomplete(line)
    autocomplete_score = score_autocomplete_string(missing_brackets)
    autocomplete_scores.append(autocomplete_score)

autocomplete_scores.sort()
middle_ind = (len(autocomplete_scores) - 1) // 2
middle_value = autocomplete_scores[middle_ind]

print(f"Solution to star 2: {middle_value}.")