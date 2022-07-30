# Read in data
with open("input19.txt") as f:
    puzzle_input = f.read()

example_input = \
"""H => HO
H => OH
O => HH

HOH"""


def parse(raw):
    transformation_string, molecule_string = raw.split("\n\n")

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


transformations, medicine = parse(puzzle_input)

new_strings = make_all_replacements(medicine, transformations)
n_unique = len(set(new_strings))
print(f"There are {n_unique} unique new molecules.")


def find_quickest_growth(start, replacements, target):
    """Starts from the target string and applies inverse transformations until we hit the starting string."""

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

        distance_to_start = string2shortest.get(start, float("inf"))
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
        # Update the list of apossible paths
        paths = sorted(newpaths, key=len) + keep

        # Print status
        msg = f"Iteration {n}, examining {len(paths)} paths."
        if start in string2shortest:
            msg += f" Shortest path found is {string2shortest[start]} steps."
        else:
            msg += " NO solution found yet."
        msg += f" Shortest string: {min(map(len, paths))}. Extended {len(extend)} paths.."
        print(msg, end="\r")
    print()

    return string2shortest[start]


n_iterations_needed = find_quickest_growth(start="e", replacements=transformations, target=medicine)
print(f"The reindeer medicine molecule can be produced in {n_iterations_needed} steps.")
