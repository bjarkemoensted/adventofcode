from collections import Counter, defaultdict


def parse(s):
    starting_polymer, instructions = s.split("\n\n")
    replacements = {}
    for line in instructions.split("\n"):
        old, insertion = line.strip().split(' -> ')
        replacement = "".join([old[0], insertion])
        replacements[old] = replacement

    return starting_polymer, replacements


def read():
    with open("input14.txt") as f:
        s = f.read()
    return parse(s)

sample = """NNCB

CH -> B
HH -> N
CB -> H
NH -> C
HB -> C
HC -> B
HN -> C
NN -> C
BH -> H
NC -> B
NB -> B
BN -> B
BB -> N
BC -> B
CC -> N
CN -> C"""



polymer, replacements = read()
polymer, replacements = parse(sample)


def grow_polymer_one_step(polym):
    newstring = ""
    for i in range(len(polym)-1):
        substring = polym[i:i+2]
        newstring += replacements[substring]

    newstring += polym[-1]
    return newstring


def get_substrings(s):
    d = {}
    for i in range(len(s) - 2):
        substring = s[i:i+2]
        d[substring] = d.get(substring, 0) + 1
    secondlast = s[-2]
    d[secondlast] = 1
    last = s[-1]
    return d, last


def grow_polymer_sneaky(s, n_iterations=10):
    substring_counts, lastchar = get_substrings(s)
    newcounts = defaultdict(lambda: 0)
    for substring, count in substring_counts.items():
        try:
            new_substring = replacements[substring]
        except KeyError:
            new_key = substring+lastchar
            new_substring = replacements[new_key]
            newcounts[new_substring[1]] = 1
        print(substring, new_substring)
        newcounts[new_substring] += count
    print(newcounts)


print(polymer)
grow_polymer_sneaky(polymer)

#for _ in range(10):
#    polymer = grow_polymer_one_step(polymer)

#counts = sorted(Counter(polymer).items(), key=lambda t: t[1])
#diff = counts[-1][1] - counts[0][1]
#print(f"Difference between most and least common element: {diff}.")


