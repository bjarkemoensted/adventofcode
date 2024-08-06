# Read in data
with open("input16.txt") as f:
    puzzle_input = f.read()


def parse(s):
    res = {}
    for line in s.split("\n"):
        sue = int(line.split(":")[0][4:])
        snip = ": ".join(line.split(": ")[1:])
        data = {}
        for part in snip.split(", "):
            k, v = part.split(": ")
            data[k] = int(v)

        res[sue] = data

    return res


forensic_analysis_str = \
"""children: 3
cats: 7
samoyeds: 2
pomeranians: 3
akitas: 0
vizslas: 0
goldfish: 5
trees: 3
cars: 2
perfumes: 1"""


# Map each compound to its quantity based on the analysis results
forensics = {c: int(q) for c, q in map(lambda s: s.split(": "), forensic_analysis_str.split("\n"))}
d = parse(puzzle_input)

# Makes list of all auntie Sues and eliminate those for whom the input is inconsistent with evidence
candidate_sues = sorted(d.keys())
for i in range(len(candidate_sues)-1, -1, -1):
    sue = candidate_sues[i]
    if any(forensics[k] != v for k, v in d[sue].items()):
        del candidate_sues[i]

assert len(candidate_sues) == 1
print(f"Only suspect left after elimination is Sue number {candidate_sues[0]}.")


# Go again but with the updated criteria
candidate_sues = sorted(d.keys())
for i in range(len(candidate_sues)-1, -1, -1):
    sue = candidate_sues[i]
    eliminate = False
    for k, v in d[sue].items():
        expected = forensics[k]
        if k in ('cats', 'trees'):
            if v <= expected:
                eliminate = True
            #
        elif k in ('pomeranians', 'goldfish'):
            if v >= expected:
                eliminate = True
            #
        elif v != expected:
            eliminate = True
    if eliminate:
        del candidate_sues[i]
    #

print(f"The remaining suspect after applying the new elimination rules is: {candidate_sues[0]}.")
