alph = "abcdefghijklmnopqrstuvwxyz"
ind2letter = {i: letter for i, letter in enumerate(alph)}
letter2ind = {letter: i for i, letter in ind2letter.items()}


def increment(s):
    res = list(s)
    for i in range(len(s)-1, -1, -1):
        char = s[i]
        ind = letter2ind[char]
        newind = (ind + 1) % len(alph)
        newchar = ind2letter[newind]
        res[i] = newchar
        if newind != 0:
            return "".join(res)


def password_is_valid(s):
    inds = [letter2ind[char] for char in s]
    forbidden = "iol"
    if any(char in forbidden for char in s):
        return False
    if not any(inds[i+2] == inds[i+1] + 1 == inds[i] + 2 for i in range(len(inds) - 2)):
        return False

    n_double_letters = 0
    i = 0
    while i < len(s) - 1:
        if s[i] == s[i+1]:
            n_double_letters += 1
            i += 2
        else:
            i += 1
        #
    if n_double_letters < 2:
        return False

    return True


def solve(data: str):
    password = data
    while not password_is_valid(password):
        password = increment(password)

    star1 = password
    print(f"Solution to part 1: {star1}")

    password = increment(password)
    while not password_is_valid(password):
        password = increment(password)

    star2 = password
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2015, 11
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
