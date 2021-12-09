import copy

digits_list = []
displays_list = []

with open("input08.txt") as f:
    for line in f.readlines():
        a, b = [elem.split() for elem in line.split(" | ")]
        digits_list.append(a)
        displays_list.append(b)
    #

digit2segments = {
    0: 'abcefg',
    1: 'cf',
    2: 'acdeg',
    3: 'acdfg',
    4: 'bcdf',
    5: 'abdfg',
    6: 'abdefg',
    7: 'acf',
    8: 'abcdefg',
    9: 'abcdfg'
}
segments2digits = {v: k for k, v in digit2segments.items()}


def get_unique_chars(strings):
    unique = set([])
    sets = [set(s) for s in strings]
    for set_ in sets:
        unique.update(set_)
    return unique


def count(strings):
    unique = get_unique_chars(strings)
    sets = [set(s) for s in strings]

    char2count = {char: sum(char in set_ for set_ in sets) for char in unique}
    return char2count


# Map each letter to the letters they must co-occur with
chars = get_unique_chars(digit2segments.values())
cooccurrences = {char: set([]) for char in sorted(chars)}
for s in digit2segments.values():
    for char in s:
        cooccurrences[char].update(set([c for c in s if c != char]))

# Map each letter to the number of times they must occur
letter2count = count(digit2segments.values())
count2possible_letters = {x: "" for x in set(letter2count.values())}
for letter, n in letter2count.items():
    count2possible_letters[n] = "".join(sorted(count2possible_letters[n] + letter))

# Count number of occurrences of digiets 1,4,7, and 8 (which have unique lengths)
target_digits = [1, 4, 7, 8]
target_lengths = {len(digit2segments[k]) for k in target_digits}

n_target = sum(len(s) in target_lengths for s in sum(displays_list, []))
print(f"Solution to star 1: {n_target}.")


def brute(locked, free):
    """Recursively generates all possible mappings from inputs."""
    locked = copy.deepcopy(locked)
    free = copy.deepcopy(free)

    # Are we done yet?
    if not free:
        yield locked

    for char in list(free.keys()):
        for candidate in sorted(free[char]):
            new_free = copy.deepcopy(free)
            for k in list(new_free.keys()):
                if k == char:
                    # Assume char maps to candidate:
                    new_free[k] = set(candidate)
                else:
                    # Assume other chars don't map to candidate
                    try:
                        new_free[k].remove(candidate)
                    except KeyError:
                        pass
                    #
                #
            # Update lists of locked and free candidates
            for k in list(new_free.keys()):
                if len(new_free[k]) == 1:
                    locked[k] = list(new_free[k])[0]
                    del new_free[k]

            for comb in brute(locked, new_free):
                yield comb


def decode(digs, mapping):
    """Decodes input digit segments (e.g. 'abc') using a mapping e.g. {'a': 'b', ...}"""
    res = []
    for dig in digs:
        decoded = "".join(sorted([mapping[char] for char in dig]))
        res.append(decoded)
    return res


def crack(digs):
    # Determine most of the mapping from segments on broken display to intended segment from number of digits the segments partake in
    counts = count(digs)
    char2candidates = {char: count2possible_letters[count] for char, count in counts.items()}
    # This is the part of the mapping which can be identified, e.g. {'a': 'b'} if a maps to b
    locked = {k: v for k, v in char2candidates.items() if len(v) == 1}
    # This is the ambiguous part, e.g. {'b': 'cd'} if b can be either c or d, based on the above
    free = {k: set(v) for k, v in char2candidates.items() if len(v) > 1}

    # Brute force the remaining combinations
    for mapping in brute(locked, free):
        decoded = decode(digs, mapping)
        if all(s in segments2digits for s in decoded):
            return mapping
        #
    #


numbers = []
for digits, displays in zip(digits_list, displays_list):
    # Find the mapping from segments on the broken display, to the real segments
    mapping = crack(digits)
    # Decode the displayed digits into the correct segments using the mapping
    decoded_segments = decode(displays, mapping)
    # Translate the decoded segments into the digits that should be displayed
    true_digits = [str(segments2digits[s]) for s in decoded_segments]
    # Convert into a four-digit number and append to results
    number = int("".join(map(str, true_digits)))
    numbers.append(number)

star2 = sum(numbers)
print(f"Solution to star 2: {star2}.")