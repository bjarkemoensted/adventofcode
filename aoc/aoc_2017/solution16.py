from string import ascii_lowercase


def parse(s):
    instructions = []
    for part in s.split(","):
        if part.startswith("s"):
            ins = ("s", int(part[1:]))
        elif part.startswith("x"):
            a, b = map(int, part[1:].split("/"))
            ins = ("x", a, b)
        elif part.startswith("p"):
            a, b = part[1:].split("/")
            ins = ("p", a, b)
        else:
            raise ValueError
        instructions.append(ins)
    return instructions


def _dance(instructions, letters):
    """Do a single dance"""

    letters = [let for let in letters]
    for op, *args in instructions:
        if op == "s":
            # Spin operation - move n elements from front to back
            i = args[0]
            letters = letters[-i:] + letters[:-i]
        elif op == "x":
            # Exchange operation - swap two elements
            a, b = args
            letters[a], letters[b] = letters[b], letters[a]
        elif op == "p":
            # Partner operation - swap two letters
            a, b = [letters.index(let) for let in args]
            letters[a], letters[b] = letters[b], letters[a]
        else:
            raise ValueError

    res = "".join(letters)
    return res


def dance(instructions, n_repeats=1):
    """Repeatedly do the program dance thingy"""

    # Determine the number of letters to use (the example uses 5 and only has a few instructions)
    n = 16 if len(instructions) > 100 else 5
    letters = list(ascii_lowercase[:n])

    # Keep track the results we've seen and the number of iterations it took to get there
    n_runs = 0
    order = "".join(letters)
    first_seen = {order: n_runs}
    strings = [order]

    while n_runs < n_repeats:
        # Do one more dance and update the letters
        order = _dance(instructions, letters)
        letters = list(order)
        n_runs += 1

        # Check for cycles in the results
        recurrence = order in first_seen
        if recurrence:
            # The remaining number of dances modulo the cycle length will change nothing
            dances_left = n_repeats - n_runs
            last_seen = first_seen[order]
            cycle_len = n_runs - last_seen
            final_ind = last_seen + dances_left % cycle_len
            order = strings[final_ind]
            break
        else:
            first_seen[order] = n_runs
            strings.append(order)

    return order


def solve(data: str):
    instructions = parse(data)

    star1 = dance(instructions)
    print(f"Solution to part 1: {star1}")

    n_repeats = 1_000_000_000 if len(instructions) > 100 else 2  # The example only uses 2 repititions
    star2 = dance(instructions, n_repeats=n_repeats)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main():
    year, day = 2017, 16
    from aoc.utils.data import check_examples
    check_examples(year=year, day=day, solver=solve)
    from aocd import get_data
    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
