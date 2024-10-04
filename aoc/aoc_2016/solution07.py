def parse(s):
    res = s.split("\n")
    return res


def substring_is_abba(string, position):
    """Checks if the string contains a pattern like 'abba' at input position."""
    try:
        a = string[position]
        b = string[position+1]
        return string[position+2] == b and string[position+3] == a and a != b
    except IndexError:
        return False


def substring_is_aba(string, position):
    """Checks if the string contains a pattern like 'aba' at input position."""
    try:
        a = string[position]
        b = string[position+1]
        return string[position+2] == a and a != b
    except IndexError:
        return False


def ip_iterator(ip_string):
    """Takes a string representing an IP address. Yields tuples of inds and whether the ind is inside brackets."""
    inside_brackets = False
    for ind, char in enumerate(ip_string):
        if char == "[":
            inside_brackets = True
            continue
        elif char == "]":
            inside_brackets = False
            continue
        yield ind, inside_brackets


def ip_supports_tls(ip_string):
    has_abba = False
    for i, inside_brackets in ip_iterator(ip_string):
        if substring_is_abba(ip_string, i):
            if inside_brackets:
                return False
            else:
                has_abba = True
            #
        #

    return has_abba


def ip_supports_ssl(ip_string):
    """Determines whether any substring like ABA outside a bracket, has a corresponding substring like BAB
    inside a bracket."""

    abas = [[], []]  # outside and inside brackets, respectively
    for i, inside_brackets in ip_iterator(ip_string):
        if substring_is_aba(ip_string, i):
            aba = ip_string[i:i+3]
            ind = int(inside_brackets)
            abas[ind].append(aba)
        #

    abas_outside, abas_inside = abas
    for aba in abas_outside:
        # Construct the complementary substring which must exist inside a bracket
        a, b, _ = aba
        bab = "".join([b, a, b])
        if bab in abas_inside:
            return True
        #

    return False


def solve(data: str):
    ip_strings = parse(data)

    star1 = sum(ip_supports_tls(ip_string) for ip_string in ip_strings)
    print(f"Solution to part 1: {star1}")

    star2 = sum(ip_supports_ssl(ip_string) for ip_string in ip_strings)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2016, 7
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
