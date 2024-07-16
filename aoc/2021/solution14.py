from collections import Counter, defaultdict


def parse(s):
    """Parses contents of input file into:
    a) starting polymer, e.g. 'NNBC',
    b) growing instructions, e.g. 'ab -> c' (which means insert c in between a's and b's),
    which is equivalent to chaning every 'ab' pair into one 'ac' and one 'cb' pair.
    Format of replacements dict is {'ab': {'ac': 1, 'cb': 1}}."""
    starting_polymer, instructions = s.split("\n\n")
    replacements = {}
    for line in instructions.split("\n"):
        # Figure out what to replacethe old string with
        old, insertion = line.strip().split(' -> ')
        pair_a = old[0] + insertion  # Make the string like 'ac'
        pair_b = insertion + old[1]  # Make the string like 'cb'
        replacement = {}
        for pair in (pair_a, pair_b):
            replacement[pair] = replacement.get(pair, 0) + 1
        replacements[old] = replacement

    return starting_polymer, replacements


def read_input():
    """Read and parse the input"""
    with open("input14.txt") as f:
        s = f.read()
    return parse(s)


polymer, replacements = read_input()


def get_substrings(s):
    """Counts the occurrences of all pairs (length 2 substrings) in input string.
    For instance, 'NNBC' -> {'NN': 1, 'NB': 1, 'BC': 1}"""
    d = {}
    for i in range(len(s) - 1):
        substring = s[i:i + 2]
        d[substring] = d.get(substring, 0) + 1
    return d


def grow_polymer_sneaky(s, n_iterations=10):
    """Grows a polymer sneakily. In each iteration, each pair (length 2 substring) grows two new pairs,
    according to the replacement data in the replacement dict.
    After n iterations, the number of occurrences of each letter is counted."""

    # Represent the polymer as all its pairs
    pairs = get_substrings(s)
    for _ in range(n_iterations):
        new_pairs = defaultdict(lambda: 0)
        # For each pair, grow its two descendants
        for pair, count in pairs.items():
            added_pairs = replacements[pair]
            for added_pair, weight in added_pairs.items():
                new_pairs[added_pair] += count * weight
            #
        pairs = new_pairs

    # Count each letter in the polymer
    counts = {}
    for substring, count in pairs.items():
        for char in substring:
            counts[char] = counts.get(char, 0) + count
        #

    # The method double counts every letter except the first and last. Compensate for this
    first, last = s[0], s[-1]
    counts[first] += 1
    counts[last] += 1
    assert all(v % 2 == 0 for v in counts.values())
    res = {k: v // 2 for k, v in counts.items()}

    return res


def compute_maxmin_diff(d):
    # Takes a dict of counts and computes the differences between the most and least common object.
    ordered = sorted(d.values())
    diff = ordered[-1] - ordered[0]
    return diff


poly_a = grow_polymer_sneaky(polymer, 10)
diff_a = compute_maxmin_diff(poly_a)
print(f"Solution to star 1: {diff_a}")

poly_b = grow_polymer_sneaky(polymer, 40)
diff_b = compute_maxmin_diff(poly_b)
print(f"Solution to star 1: {diff_b}")