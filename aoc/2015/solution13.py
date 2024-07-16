from itertools import permutations

# Read in data
with open("input13.txt") as f:
    puzzle_input = f.read()

example_input = \
"""Alice would gain 54 happiness units by sitting next to Bob.
Alice would lose 79 happiness units by sitting next to Carol.
Alice would lose 2 happiness units by sitting next to David.
Bob would gain 83 happiness units by sitting next to Alice.
Bob would lose 7 happiness units by sitting next to Carol.
Bob would lose 63 happiness units by sitting next to David.
Carol would lose 62 happiness units by sitting next to Alice.
Carol would gain 60 happiness units by sitting next to Bob.
Carol would gain 55 happiness units by sitting next to David.
David would gain 46 happiness units by sitting next to Alice.
David would lose 7 happiness units by sitting next to Bob.
David would gain 41 happiness units by sitting next to Carol."""


def parse(s):
    """Parses into e.g. {'Alice': {"Bob": -42, ...}, ...}"""
    res = {}
    for line in s.split("\n"):
        words = line.split(" ")
        a = words[0]
        b = words[-1].replace('.', '')
        score = int(words[3]) * (-1 if words[2] == "lose" else 1)
        if a in res:
            res[a][b] = score
        else:
            res[a] = {b: score}

    return res


def compute_happiness(arrangement, prefdict):
    score = 0
    for i, name in enumerate(arrangement):
        neighbors = [arrangement[(i+inc) % len(arrangement)] for inc in (-1, 1)]
        person_prefs = prefdict.get(name, {})
        contribution = sum(person_prefs.get(neighbor, 0) for neighbor in neighbors)
        score += contribution

    return score


def determine_optimum_arrangement(names, preferences):
    res = float('-inf')

    for arrangement in permutations(names):
        happiness = compute_happiness(arrangement, preferences)
        res = max(res, happiness)

    return res


prefs = parse(puzzle_input)
names = sorted(prefs.keys())
max_score = determine_optimum_arrangement(names=names, preferences=prefs)

print(f"Maximum felicity is {max_score}.")

names_with_me = [name for name in names] + ["Me"]
new_max = determine_optimum_arrangement(names=names_with_me, preferences=prefs)
print(f"Happiness by including me: {new_max}.")