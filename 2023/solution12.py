from functools import cache


def read_input():
    with open("input12.txt") as f:
        puzzle_input = f.read()
    return puzzle_input


def parse(s):
    res = []
    for line in s.split("\n"):
        a, b = line.split()
        nums = tuple(int(elem) for elem in b.split(","))
        res.append((a, nums))

    return res


@cache
def meh(springs, code):
    """Returns the number of combinations of working (.) and broken (#) that satisfy the group criteria specified
    by the input code, e.g. (3,2,4) means 3 groups of neighboring, broken springs with 3, 2, and 4 in each,
    respectively."""

    res = 0
    done = code == ()
    if not springs:
        # If no springs, we're happy iff there's also no more requirements
        res = int(done)
    elif done:
        # If springs and no more requirements, we're happy iff no springs are broken
        res = int("#" not in springs)
    else:
        # If we're not done yet, determine whether a new group starts, the current is continued, or both (for '?')
        char = springs[0]
        continue_group = char in ".?"
        break_group = char in "#?"

        if continue_group:
            # The spring works (or we assume so for ?). Look at the remaining n - 1 springs.
            newsprings = springs[1:]
            newcode = code
            res += meh(newsprings, newcode)
        if break_group:
            # This spring is broken (or we assume so). If consistent with code, recurse with remaining code+springs

            n_broken_needed = code[0]
            # If the current broken group ends too early ("." before n_broken_needed) or too long, we're done
            group_too_short = "." in springs[:n_broken_needed]
            group_too_long = len(springs) > n_broken_needed and springs[n_broken_needed] == "#"

            # We're also done if n_springs is smaller than the required number of broken springs
            too_few_springs = len(springs) < n_broken_needed

            # If we're not done here, the first of the remaining rules is satisfied. Recurse with the rest.
            recurse = not any(cond for cond in (group_too_long, group_too_short, too_few_springs))
            if recurse:
                newsprings = springs[n_broken_needed+1:]
                newcode = code[1:]
                res += meh(newsprings, newcode)
            #
        #

    return res


def unfold(data):
    res = []
    for springs, code in data:
        springs_new = "?".join(5*[springs])
        code_new = tuple(5*list(code))
        res.append((springs_new, code_new))

    return res


def count_possible_arrangements(data):
    res = 0

    for springs, code in data:
        res += meh(springs, code)

    return res


def main():
    raw = read_input()
    data = parse(raw)

    star1 = count_possible_arrangements(data)
    print(star1)

    data2 = unfold(data)
    star2 = count_possible_arrangements(data2)
    print(star2)


if __name__ == '__main__':
    main()
