# · +· .`·   ·+.   · `   *  ` ·. ·  · *`   .·`  ·*·  `.·   .·   ·• ·    .*`··  +
# `·. ·`    ·   · ·`* .  +·    Medicine for Rudolph `·    .· *` • ·.  ·  · `. ·.
# ·.`.  · ·+   ·  `·   https://adventofcode.com/2015/day/19 . ·    `·*. +    .`·
# .`·  + ·   ·.      ·* ·.`   + ·`    ·. +·  `· ·.    ·   .``· + · `   ··..* · `


def parse(s: str):
    transformation_string, molecule_string = s.split("\n\n")

    replacements = []
    for line in transformation_string.split("\n"):
        old, new = line.split(" => ")
        replacements.append((old, new))

    return replacements, molecule_string


def replace_substring(string, old_substring, new_substring, index):
    """Replaces target substring at the specified index, e.g.
    'abc', 'b', 'fe' -> 'afec'."""
    pre_snip = string[:index]
    post_snip = string[index+len(old_substring):]
    res = pre_snip + new_substring + post_snip

    return res


def make_all_replacements(string, replacements):
    """Generates all the new strings that can be generated from the input string and the input replacements."""
    res = []
    for i in range(len(string)):
        for old, new in replacements:
            if string[i:].startswith(old):
                transformed = replace_substring(
                    string=string,
                    old_substring=old,
                    new_substring=new,
                    index=i)
                res.append(transformed)
            #
        #
    return res


def find_quickest_growth(start, replacements, target, maxiter=None):
    """Starts from the target string and applies inverse transformations until we hit the starting string."""

    if maxiter is None:
        maxiter = float("inf")
    string2shortest = {target: 0}
    paths = [target]
    n = 0
    # We start from the 'target' molecule and iterate towards the start, so we must apply inverse transformations
    inverse_transformations = [(b, a) for a, b in replacements]

    while start not in string2shortest:
        # While no solution, keep trying to reduce the 100 shortest strings
        n += 1
        newpaths = []
        cut = 100
        extend = paths[:cut]
        keep = paths[cut:]

        for string in extend:
            # Try extending the shortest string using the inverse transformations
            dist = string2shortest[string]
            batch = make_all_replacements(string, inverse_transformations)
            newdist = dist + 1
            for newstring in batch:
                # There's hope for the new string if the string is new or we found a quicker way to it
                hope = newdist < string2shortest.get(newstring, float('inf'))
                if hope:
                    string2shortest[newstring] = newdist
                    newpaths.append(newstring)
                #
            #
        # Update the list of possible paths
        paths = sorted(newpaths, key=len) + keep

        # Print status
        msg = f"Iteration {n}, examining {len(paths)} paths."
        if start in string2shortest:
            msg += f" Shortest path found is {string2shortest[start]} steps."
        else:
            msg += " NO solution found yet."
        msg += f" Shortest string: {min(map(len, paths)) if paths else 'n/a'}. Extended {len(extend)} paths.."
        print(msg, end="\r")

        if n >= maxiter:
            break

    print()

    try:
        res = string2shortest[start]
    except KeyError:
        res = None

    return res


def solve(data: str) -> tuple[int|str, int|str]:
    transformations, medicine = parse(data)
    if not any(a == "e" for a, _ in transformations):
        transformations += [('e', 'H'), ('e', 'O')]

    new_strings = make_all_replacements(medicine, transformations)
    star1 = len(set(new_strings))
    print(f"Solution to part 1: {star1}")

    star2 = find_quickest_growth(start="e", replacements=transformations, target=medicine)
    print(f"Solution to part 2: {star2}")

    return star1, star2


def main() -> None:
    year, day = 2015, 19
    from aocd import get_data

    raw = get_data(year=year, day=day)
    solve(raw)


if __name__ == '__main__':
    main()
